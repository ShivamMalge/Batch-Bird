//! SQL text -> [`LogicalPlan`].
//!
//! `sqlparser-rs` does the lexing and grammar (`techstack.md`: we do not hand-roll a
//! parser). This module does the part that is actually ours: deciding what the engine will
//! and will not execute.
//!
//! # Rejection is the feature
//! Roughly one line here builds a plan and the rest refuse to. That ratio is intentional.
//! `sqlparser` happily accepts window functions, CTEs, `GROUP BY CUBE`, and joins -- and an
//! engine that silently ignored the clauses it did not implement would return confidently
//! wrong answers. Every clause the supported shape does not mention is matched and rejected
//! by name, so an out-of-scope query fails at parse time with a message that says which
//! construct was the problem.
//!
//! Two error types keep the distinction a user actually cares about ([`Error`]):
//! [`Error::Sql`] means the text is not valid SQL, [`Error::Unsupported`] means it is valid
//! SQL that Batchbird deliberately does not do.
//!
//! # Identifiers are case-sensitive
//! **Not specified in the design docs.** Standard SQL folds unquoted identifiers, but column
//! names here have to match CSV header text, and inventing a folding rule would mean picking
//! which case wins and then applying it consistently through storage. Names are taken exactly
//! as written, so `GROUP BY Region` does not match `SELECT region`. Quoting still works and
//! is stripped: `"region"` and `region` are the same name.

use sqlparser::ast::{
    BinaryOperator, Expr, Function, FunctionArg, FunctionArgExpr, FunctionArgumentList,
    FunctionArguments, GroupByExpr, ObjectName, ObjectNamePart, Query, Select, SelectItem, SetExpr,
    Statement, TableFactor, TableWithJoins, UnaryOperator, Value,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

use crate::error::{Error, Result};
use crate::plan::{AggFunc, Aggregation, CompareOp, Literal, LogicalPlan, Predicate};

/// Shorthand for the rejection path, which is most of this module.
fn unsupported<T>(what: impl Into<String>) -> Result<T> {
    Err(Error::Unsupported(what.into()))
}

/// Parse one SQL statement into a [`LogicalPlan`], or explain why it cannot be run.
///
/// ```
/// use batchbird::parser::parse;
/// use batchbird::plan::{AggFunc, CompareOp, Literal};
///
/// let plan = parse("SELECT region, SUM(amount) FROM sales WHERE price > 10 GROUP BY region")?;
/// assert_eq!(plan.table, "sales");
/// assert_eq!(plan.group_by, "region");
/// assert_eq!(plan.aggregation.func, AggFunc::Sum);
/// assert_eq!(plan.aggregation.input, "amount");
/// assert_eq!(plan.filter.op, CompareOp::Gt);
/// assert_eq!(plan.filter.literal, Literal::Int64(10));
/// # Ok::<(), batchbird::error::Error>(())
/// ```
///
/// Anything outside the supported shape is an [`Error::Unsupported`] naming the construct:
///
/// ```
/// # use batchbird::parser::parse;
/// let err = parse("SELECT a, SUM(b) FROM t JOIN u ON t.id = u.id WHERE c > 1 GROUP BY a")
///     .unwrap_err();
/// assert!(err.to_string().contains("JOIN"));
/// ```
pub fn parse(sql: &str) -> Result<LogicalPlan> {
    let dialect = GenericDialect {};
    let mut statements = Parser::parse_sql(&dialect, sql)?;

    match statements.len() {
        1 => {}
        0 => return unsupported("empty input, with no SQL statement"),
        n => return unsupported(format!("{n} statements; exactly one query is supported")),
    }

    let query = match statements.remove(0) {
        Statement::Query(query) => *query,
        _ => return unsupported("a non-SELECT statement; only SELECT queries are supported"),
    };

    select_to_plan(query_to_select(query)?)
}

/// Strip the query wrapper, rejecting every clause that can hang off it.
fn query_to_select(query: Query) -> Result<Select> {
    // Destructured by name rather than matched with `..` so that a future `sqlparser` upgrade
    // introducing a new clause fails to compile here. A silently-ignored new clause is
    // exactly the wrong-answer bug this module exists to prevent.
    let Query {
        with,
        body,
        order_by,
        limit_clause,
        fetch,
        locks,
        for_clause,
        settings,
        format_clause,
        pipe_operators,
    } = query;

    if with.is_some() {
        return unsupported("WITH / common table expressions");
    }
    if order_by.is_some() {
        return unsupported("ORDER BY; GROUP BY output has no guaranteed order here");
    }
    if limit_clause.is_some() {
        return unsupported("LIMIT / OFFSET");
    }
    if fetch.is_some() {
        return unsupported("FETCH");
    }
    if !locks.is_empty() {
        return unsupported("row locking clauses (FOR UPDATE / FOR SHARE)");
    }
    if for_clause.is_some() {
        return unsupported("a FOR clause");
    }
    if settings.is_some() {
        return unsupported("SETTINGS");
    }
    if format_clause.is_some() {
        return unsupported("FORMAT");
    }
    if !pipe_operators.is_empty() {
        return unsupported("pipe operators");
    }

    match *body {
        SetExpr::Select(select) => Ok(*select),
        SetExpr::Query(_) => unsupported("a nested query"),
        SetExpr::SetOperation { .. } => unsupported("UNION / INTERSECT / EXCEPT"),
        _ => unsupported("this query form; only a plain SELECT is supported"),
    }
}

fn select_to_plan(select: Select) -> Result<LogicalPlan> {
    // Exhaustive, like `query_to_select` -- no `..`. `Select` is the larger struct and the
    // likelier place for a new *semantic* clause to appear in a future sqlparser, so it is
    // the one that most needs the tripwire: adding a field upstream breaks this line rather
    // than silently producing a wrong answer for a clause nobody handled.
    //
    // The four bound-and-ignored fields are syntax metadata, not semantics: where the SELECT
    // token sat, whether TOP preceded DISTINCT, whether WINDOW preceded QUALIFY, and whether
    // the query was written FROM-first. None of them changes what the query computes.
    let Select {
        distinct,
        top,
        into,
        projection,
        from,
        lateral_views,
        prewhere,
        selection,
        connect_by,
        group_by,
        cluster_by,
        distribute_by,
        sort_by,
        having,
        named_window,
        qualify,
        value_table_mode,
        optimizer_hints,
        select_modifiers,
        exclude,
        select_token: _,
        top_before_distinct: _,
        window_before_qualify: _,
        flavor: _,
    } = select;

    if distinct.is_some() {
        return unsupported("DISTINCT");
    }
    if top.is_some() {
        return unsupported("TOP");
    }
    if into.is_some() {
        return unsupported("SELECT INTO");
    }
    if !lateral_views.is_empty() {
        return unsupported("LATERAL VIEW");
    }
    if prewhere.is_some() {
        return unsupported("PREWHERE");
    }
    if !connect_by.is_empty() {
        return unsupported("CONNECT BY");
    }
    if !cluster_by.is_empty() {
        return unsupported("CLUSTER BY");
    }
    if !distribute_by.is_empty() {
        return unsupported("DISTRIBUTE BY");
    }
    if !sort_by.is_empty() {
        return unsupported("SORT BY");
    }
    if having.is_some() {
        return unsupported("HAVING");
    }
    if !named_window.is_empty() {
        return unsupported("WINDOW");
    }
    if qualify.is_some() {
        return unsupported("QUALIFY");
    }
    if value_table_mode.is_some() {
        return unsupported("SELECT AS STRUCT / SELECT AS VALUE");
    }
    if !optimizer_hints.is_empty() {
        return unsupported("optimizer hints");
    }
    if select_modifiers.is_some() {
        return unsupported("MySQL SELECT modifiers");
    }
    if exclude.is_some() {
        return unsupported("EXCLUDE");
    }

    let table = parse_from(from)?;
    let (select_column, aggregation) = parse_projection(projection)?;
    let group_by = parse_group_by(group_by)?;

    // SQL requires every non-aggregated SELECT column to appear in GROUP BY. With exactly one
    // group column and one plain SELECT column, "appears in" collapses to "is equal to".
    if select_column != group_by {
        return unsupported(format!(
            "SELECT column {select_column:?} with GROUP BY {group_by:?}; \
             they must be the same column (identifiers are case-sensitive here)"
        ));
    }

    let filter = parse_where(selection)?;

    Ok(LogicalPlan {
        table,
        group_by,
        aggregation,
        filter,
    })
}

/// `FROM t` -- exactly one plain table, no joins.
fn parse_from(mut from: Vec<TableWithJoins>) -> Result<String> {
    match from.len() {
        1 => {}
        0 => return unsupported("a query with no FROM clause"),
        n => {
            return unsupported(format!(
                "{n} tables in FROM (an implicit cross join); joins are an explicit non-goal"
            ));
        }
    }

    let TableWithJoins { relation, joins } = from.remove(0);
    if !joins.is_empty() {
        // agents.md Hard Guardrail: joins need explicit human sign-off, and join execution
        // plus join ordering is scoped as a separate project (prd.md Non-Goals).
        return unsupported("a JOIN; joins are an explicit non-goal of this engine");
    }

    match relation {
        TableFactor::Table {
            name, alias, args, ..
        } => {
            if alias.is_some() {
                return unsupported("a table alias");
            }
            if args.is_some() {
                return unsupported("a table-valued function");
            }
            object_name(name)
        }
        TableFactor::Derived { .. } => unsupported("a subquery in FROM"),
        TableFactor::NestedJoin { .. } => {
            unsupported("a JOIN; joins are an explicit non-goal of this engine")
        }
        _ => unsupported("this FROM form; only a plain table name is supported"),
    }
}

/// Flatten a possibly-qualified name, rejecting the qualified case.
fn object_name(name: ObjectName) -> Result<String> {
    let ObjectName(mut parts) = name;
    if parts.len() != 1 {
        return unsupported("a qualified name like schema.table");
    }
    match parts.remove(0) {
        ObjectNamePart::Identifier(ident) => Ok(ident.value),
        _ => unsupported("a function-style name part"),
    }
}

/// `SELECT col1, SUM(col2)` -- exactly two items, in that order.
///
/// The order is fixed rather than sniffed. `SELECT SUM(x), region ...` is perfectly sensible
/// SQL, but accepting it widens the supported surface past the one shape in `prd.md`, and
/// widening needs sign-off (`agents.md`).
fn parse_projection(projection: Vec<SelectItem>) -> Result<(String, Aggregation)> {
    // Wildcards are checked before the item count, because `SELECT *` is one item and
    // "a SELECT list of 1 item(s)" would be a uselessly indirect way to say "no wildcards".
    if projection.iter().any(|item| {
        matches!(
            item,
            SelectItem::Wildcard(_) | SelectItem::QualifiedWildcard(..)
        )
    }) {
        return unsupported("a wildcard (SELECT *); the columns must be named");
    }

    if projection.len() != 2 {
        return unsupported(format!(
            "a SELECT list of {} item(s); exactly two are required, the group column then SUM(col)",
            projection.len()
        ));
    }

    let mut items = projection.into_iter();
    let column = select_item_column(items.next().expect("length checked"))?;
    let aggregation = select_item_aggregation(items.next().expect("length checked"))?;
    Ok((column, aggregation))
}

fn select_item_column(item: SelectItem) -> Result<String> {
    match item {
        SelectItem::UnnamedExpr(Expr::Identifier(ident)) => Ok(ident.value),
        SelectItem::UnnamedExpr(Expr::CompoundIdentifier(_)) => {
            unsupported("a qualified column name like t.col")
        }
        SelectItem::UnnamedExpr(Expr::Function(_)) => unsupported(
            "an aggregate as the first SELECT item; the shape is SELECT col1, SUM(col2)",
        ),
        SelectItem::UnnamedExpr(_) => {
            unsupported("an expression as the first SELECT item; only a plain column works")
        }
        SelectItem::ExprWithAlias { .. } | SelectItem::ExprWithAliases { .. } => {
            unsupported("a column alias (AS)")
        }
        _ => unsupported("a wildcard (SELECT *)"),
    }
}

fn select_item_aggregation(item: SelectItem) -> Result<Aggregation> {
    let expr = match item {
        SelectItem::UnnamedExpr(expr) => expr,
        SelectItem::ExprWithAlias { .. } | SelectItem::ExprWithAliases { .. } => {
            return unsupported("a column alias (AS)");
        }
        _ => return unsupported("a wildcard (SELECT *)"),
    };

    let function = match expr {
        Expr::Function(function) => function,
        Expr::Identifier(_) | Expr::CompoundIdentifier(_) => unsupported(
            "a plain column as the second SELECT item; the shape is SELECT col1, SUM(col2)",
        )?,
        _ => return unsupported("an expression as the second SELECT item; expected SUM(col)"),
    };

    parse_sum(function)
}

fn parse_sum(function: Function) -> Result<Aggregation> {
    let Function {
        name,
        uses_odbc_syntax,
        parameters,
        args,
        filter,
        null_treatment,
        over,
        within_group,
    } = function;

    let name = object_name(name)?;
    if !name.eq_ignore_ascii_case("SUM") {
        // systemDesign.md keeps the Accumulator trait precisely so this is a scope decision
        // rather than an architectural limit -- AVG/COUNT would be new accumulators, not a
        // new engine.
        return unsupported(format!(
            "the {} aggregate; only SUM is implemented",
            name.to_uppercase()
        ));
    }

    if uses_odbc_syntax {
        return unsupported("ODBC function syntax");
    }
    if !matches!(parameters, FunctionArguments::None) {
        return unsupported("a parameterised aggregate");
    }
    if filter.is_some() {
        return unsupported("FILTER on an aggregate");
    }
    if null_treatment.is_some() {
        return unsupported("IGNORE NULLS / RESPECT NULLS; the engine has no NULL representation");
    }
    if over.is_some() {
        return unsupported("a window function (OVER)");
    }
    if !within_group.is_empty() {
        return unsupported("WITHIN GROUP");
    }

    let FunctionArgumentList {
        duplicate_treatment,
        args,
        clauses,
    } = match args {
        FunctionArguments::List(list) => list,
        FunctionArguments::None => return unsupported("SUM with no argument"),
        FunctionArguments::Subquery(_) => return unsupported("a subquery inside SUM"),
    };

    if duplicate_treatment.is_some() {
        return unsupported("SUM(DISTINCT ...)");
    }
    if !clauses.is_empty() {
        return unsupported("argument clauses inside the aggregate call");
    }
    if args.len() != 1 {
        return unsupported(format!("SUM with {} arguments; it takes one", args.len()));
    }

    let input = match args.into_iter().next().expect("length checked") {
        FunctionArg::Unnamed(FunctionArgExpr::Expr(Expr::Identifier(ident))) => ident.value,
        FunctionArg::Unnamed(FunctionArgExpr::Expr(Expr::CompoundIdentifier(_))) => {
            return unsupported("a qualified column name like t.col");
        }
        FunctionArg::Unnamed(FunctionArgExpr::Expr(_)) => {
            return unsupported("an expression inside SUM; only SUM(column) is supported");
        }
        FunctionArg::Named { .. } | FunctionArg::ExprNamed { .. } => {
            return unsupported("a named function argument");
        }
        _ => return unsupported("SUM(*)"),
    };

    Ok(Aggregation {
        func: AggFunc::Sum,
        input,
    })
}

/// `GROUP BY col1` -- exactly one plain column.
fn parse_group_by(group_by: GroupByExpr) -> Result<String> {
    let exprs = match group_by {
        GroupByExpr::Expressions(exprs, modifiers) => {
            if !modifiers.is_empty() {
                return unsupported("GROUP BY modifiers (CUBE / ROLLUP / GROUPING SETS)");
            }
            exprs
        }
        GroupByExpr::All(_) => return unsupported("GROUP BY ALL"),
    };

    match exprs.len() {
        1 => {}
        // sqlparser represents a missing GROUP BY as an empty expression list.
        0 => return unsupported("a query with no GROUP BY; the supported shape always groups"),
        n => {
            // agents.md Hard Guardrail: GroupKey stays a single-value newtype. Multi-column
            // keys would need heterogeneous hashing and equality -- trait objects or
            // macro-generated tuple impls -- for a feature that is an explicit non-goal
            // (systemDesign.md "Group-By").
            return unsupported(format!(
                "GROUP BY over {n} columns; only single-column GROUP BY is supported"
            ));
        }
    }

    match exprs.into_iter().next().expect("length checked") {
        Expr::Identifier(ident) => Ok(ident.value),
        Expr::CompoundIdentifier(_) => unsupported("a qualified column name like t.col"),
        // sqlparser reports these as expressions in the GROUP BY list rather than as
        // `GroupByWithModifier`s, so the modifier check above never sees them.
        Expr::Cube(_) | Expr::Rollup(_) | Expr::GroupingSets(_) => {
            unsupported("GROUP BY modifiers (CUBE / ROLLUP / GROUPING SETS)")
        }
        _ => unsupported("an expression in GROUP BY; only a plain column is supported"),
    }
}

/// `WHERE col <op> literal` -- one comparison, no boolean combinators.
fn parse_where(selection: Option<Expr>) -> Result<Predicate> {
    let Some(expr) = selection else {
        return unsupported("a query with no WHERE clause; the supported shape always filters");
    };

    let (left, op, right) = match expr {
        Expr::BinaryOp { left, op, right } => (left, op, right),
        Expr::Nested(_) => return unsupported("a parenthesised WHERE expression"),
        Expr::Between { .. } => return unsupported("BETWEEN"),
        Expr::InList { .. } | Expr::InSubquery { .. } => return unsupported("IN"),
        Expr::Like { .. } | Expr::ILike { .. } => return unsupported("LIKE"),
        Expr::IsNull(_) | Expr::IsNotNull(_) => {
            return unsupported("IS NULL / IS NOT NULL; the engine has no NULL representation");
        }
        Expr::UnaryOp {
            op: UnaryOperator::Not,
            ..
        } => return unsupported("NOT"),
        _ => return unsupported("this WHERE form; the shape is WHERE col <op> literal"),
    };

    // The operator is matched *first*. `c > 1 AND d < 2` parses as AND at the top with a
    // comparison on each side, so inspecting the left operand first would report the
    // uninformative "an expression on the left" instead of naming AND as the problem.
    let op = match op {
        BinaryOperator::Eq => CompareOp::Eq,
        BinaryOperator::Lt => CompareOp::Lt,
        BinaryOperator::Gt => CompareOp::Gt,
        BinaryOperator::And | BinaryOperator::Or => {
            return unsupported("a compound WHERE clause (AND / OR)");
        }
        other => {
            // >=, <=, != and friends are each another SIMD comparison kernel to write and
            // benchmark in Phase 5, and demonstrate nothing the supported three do not.
            return unsupported(format!(
                "the {other} operator; only =, <, and > are supported"
            ));
        }
    };

    let column = match *left {
        Expr::Identifier(ident) => ident.value,
        Expr::CompoundIdentifier(_) => return unsupported("a qualified column name like t.col"),
        Expr::Value(_) => {
            return unsupported("a literal on the left of the comparison; write col <op> literal");
        }
        _ => return unsupported("an expression on the left of the comparison"),
    };

    let literal = parse_literal(*right)?;
    Ok(Predicate {
        column,
        op,
        literal,
    })
}

fn parse_literal(expr: Expr) -> Result<Literal> {
    match expr {
        Expr::Value(value) => value_to_literal(value.value, false),
        // A negative threshold arrives as unary minus over a positive literal, not as a
        // negative number token.
        Expr::UnaryOp {
            op: UnaryOperator::Minus,
            expr,
        } => match *expr {
            Expr::Value(value) => value_to_literal(value.value, true),
            _ => unsupported("a negated expression as the comparison value"),
        },
        Expr::UnaryOp {
            op: UnaryOperator::Plus,
            expr,
        } => match *expr {
            Expr::Value(value) => value_to_literal(value.value, false),
            _ => unsupported("a signed expression as the comparison value"),
        },
        Expr::Identifier(_) | Expr::CompoundIdentifier(_) => unsupported(
            "a column on the right of the comparison; column-to-column comparison is out of scope",
        ),
        Expr::Function(_) => unsupported("a function call as the comparison value"),
        _ => unsupported("this comparison value; the right-hand side must be a literal"),
    }
}

fn value_to_literal(value: Value, negated: bool) -> Result<Literal> {
    match value {
        Value::Number(text, _) => {
            // Re-attach the sign as text rather than negating after parsing, so i64::MIN
            // survives -- its absolute value does not fit in an i64.
            let text = if negated { format!("-{text}") } else { text };

            if let Ok(v) = text.parse::<i64>() {
                return Ok(Literal::Int64(v));
            }
            match text.parse::<f64>() {
                // Same finiteness rule the CSV loader applies: NaN would break the equality
                // that grouping depends on, so it never enters the engine.
                Ok(v) if v.is_finite() => Ok(Literal::Float64(v)),
                _ => unsupported(format!("the numeric literal {text:?}")),
            }
        }
        Value::SingleQuotedString(s) => {
            if negated {
                return unsupported("a negated string literal");
            }
            Ok(Literal::Utf8(s))
        }
        Value::Boolean(_) => {
            unsupported("a boolean literal; there is no Bool column type in this engine")
        }
        Value::Null => unsupported("NULL; the engine has no NULL representation"),
        other => unsupported(format!("the literal {other}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHAPE: &str = "SELECT region, SUM(amount) FROM sales WHERE price > 10 GROUP BY region";

    fn plan(sql: &str) -> LogicalPlan {
        parse(sql).unwrap_or_else(|e| panic!("expected {sql:?} to parse, got: {e}"))
    }

    /// Returns the rejection reason, asserting it was a scope rejection rather than a
    /// syntax error -- the two mean different things to a user.
    fn rejection(sql: &str) -> String {
        match parse(sql) {
            Err(Error::Unsupported(what)) => what,
            Err(other) => panic!("expected Unsupported for {sql:?}, got: {other:?}"),
            Ok(plan) => panic!("expected {sql:?} to be rejected, but it parsed as {plan:?}"),
        }
    }

    #[test]
    fn parses_the_supported_shape() {
        let plan = plan(SHAPE);
        assert_eq!(plan.table, "sales");
        assert_eq!(plan.group_by, "region");
        assert_eq!(plan.aggregation.func, AggFunc::Sum);
        assert_eq!(plan.aggregation.input, "amount");
        assert_eq!(plan.filter.column, "price");
        assert_eq!(plan.filter.op, CompareOp::Gt);
        assert_eq!(plan.filter.literal, Literal::Int64(10));
    }

    #[test]
    fn accepts_all_three_comparison_operators() {
        for (sql_op, expected) in [
            (">", CompareOp::Gt),
            ("<", CompareOp::Lt),
            ("=", CompareOp::Eq),
        ] {
            let sql = format!("SELECT a, SUM(b) FROM t WHERE c {sql_op} 1 GROUP BY a");
            assert_eq!(plan(&sql).filter.op, expected);
        }
    }

    #[test]
    fn accepts_int_float_and_string_literals() {
        let cases = [
            ("1", Literal::Int64(1)),
            ("-7", Literal::Int64(-7)),
            ("2.5", Literal::Float64(2.5)),
            ("-2.5", Literal::Float64(-2.5)),
            ("'north'", Literal::Utf8("north".to_string())),
        ];
        for (text, expected) in cases {
            let sql = format!("SELECT a, SUM(b) FROM t WHERE c = {text} GROUP BY a");
            assert_eq!(plan(&sql).filter.literal, expected, "literal {text}");
        }
    }

    #[test]
    fn keywords_are_case_insensitive() {
        let plan = plan("select region, sum(amount) from sales where price > 10 group by region");
        assert_eq!(plan.aggregation.func, AggFunc::Sum);
        assert_eq!(plan.table, "sales");
    }

    #[test]
    fn quoted_identifiers_lose_their_quotes() {
        let plan = plan(
            r#"SELECT "region", SUM("amount") FROM "sales" WHERE "price" > 10 GROUP BY "region""#,
        );
        assert_eq!(plan.table, "sales");
        assert_eq!(plan.group_by, "region");
        assert_eq!(plan.aggregation.input, "amount");
        assert_eq!(plan.filter.column, "price");
    }

    #[test]
    fn column_identifiers_are_case_sensitive() {
        // Documented deviation from standard SQL folding -- see the module docs. Column names
        // must match CSV header text, so no case is silently preferred.
        let what = rejection("SELECT region, SUM(b) FROM t WHERE c > 1 GROUP BY REGION");
        assert!(what.contains("case-sensitive"), "message was: {what}");
    }

    #[test]
    fn malformed_sql_is_a_syntax_error_not_a_scope_rejection() {
        // The distinction the two error variants exist for. Note that plausible-looking
        // nonsense like `SELECT FROM WHERE` is NOT a syntax error to sqlparser -- it reads
        // as a query with an empty projection -- so this uses an unclosed paren instead.
        match parse("SELECT a, SUM(b FROM t GROUP BY a") {
            Err(Error::Sql(_)) => {}
            other => panic!("expected a syntax error, got {other:?}"),
        }
    }

    #[test]
    fn every_rejection_message_names_the_supported_shape() {
        let rendered = parse("SELECT * FROM t").unwrap_err().to_string();
        assert!(rendered.contains("SELECT col1, SUM(col2)"), "{rendered}");
        assert!(rendered.contains("GROUP BY col1"), "{rendered}");
    }

    // ---- Hard guardrails (agents.md) -------------------------------------------------

    #[test]
    fn rejects_joins() {
        for sql in [
            "SELECT a, SUM(b) FROM t JOIN u ON t.id = u.id WHERE c > 1 GROUP BY a",
            "SELECT a, SUM(b) FROM t LEFT JOIN u ON t.id = u.id WHERE c > 1 GROUP BY a",
            "SELECT a, SUM(b) FROM t, u WHERE c > 1 GROUP BY a",
        ] {
            let what = rejection(sql);
            assert!(
                what.contains("JOIN") || what.contains("cross join"),
                "message for {sql:?} was: {what}"
            );
        }
    }

    #[test]
    fn rejects_multi_column_group_by() {
        let what = rejection("SELECT a, SUM(b) FROM t WHERE c > 1 GROUP BY a, d");
        assert!(what.contains("2 columns"), "message was: {what}");
        assert!(what.contains("single-column"), "message was: {what}");
    }

    #[test]
    fn rejects_aggregates_other_than_sum() {
        for func in ["COUNT", "AVG", "MIN", "MAX"] {
            let sql = format!("SELECT a, {func}(b) FROM t WHERE c > 1 GROUP BY a");
            let what = rejection(&sql);
            assert!(what.contains(func), "message for {func} was: {what}");
            assert!(what.contains("only SUM"), "message for {func} was: {what}");
        }
    }

    // ---- Clause-by-clause rejections ------------------------------------------------

    #[test]
    fn rejects_out_of_scope_clauses() {
        let cases = [
            (
                "SELECT a, SUM(b) FROM t WHERE c > 1 GROUP BY a HAVING SUM(b) > 2",
                "HAVING",
            ),
            (
                "SELECT a, SUM(b) FROM t WHERE c > 1 GROUP BY a ORDER BY a",
                "ORDER BY",
            ),
            (
                "SELECT a, SUM(b) FROM t WHERE c > 1 GROUP BY a LIMIT 5",
                "LIMIT",
            ),
            (
                "SELECT DISTINCT a, SUM(b) FROM t WHERE c > 1 GROUP BY a",
                "DISTINCT",
            ),
            (
                "WITH x AS (SELECT 1) SELECT a, SUM(b) FROM t WHERE c > 1 GROUP BY a",
                "WITH",
            ),
            (
                "SELECT a, SUM(b) FROM t WHERE c > 1 GROUP BY a UNION SELECT a, SUM(b) FROM t WHERE c > 1 GROUP BY a",
                "UNION",
            ),
            (
                "SELECT a, SUM(b) FROM (SELECT * FROM u) WHERE c > 1 GROUP BY a",
                "subquery",
            ),
            (
                "SELECT a, SUM(b) OVER () FROM t WHERE c > 1 GROUP BY a",
                "window function",
            ),
            (
                "SELECT a, SUM(DISTINCT b) FROM t WHERE c > 1 GROUP BY a",
                "DISTINCT",
            ),
            (
                "SELECT a, SUM(b) AS total FROM t WHERE c > 1 GROUP BY a",
                "alias",
            ),
            (
                "SELECT a, SUM(b) FROM t AS x WHERE c > 1 GROUP BY a",
                "alias",
            ),
            (
                "SELECT a, SUM(b) FROM t WHERE c > 1 GROUP BY CUBE (a)",
                "GROUP BY modifiers",
            ),
            (
                "SELECT a, SUM(b) FROM t WHERE c > 1 GROUP BY ALL",
                "GROUP BY ALL",
            ),
            ("INSERT INTO t VALUES (1)", "non-SELECT"),
        ];

        for (sql, expected) in cases {
            let what = rejection(sql);
            assert!(
                what.contains(expected),
                "for {sql:?}\n  expected message containing {expected:?}\n  got: {what}"
            );
        }
    }

    #[test]
    fn rejects_malformed_projections() {
        let cases = [
            ("SELECT * FROM t WHERE c > 1 GROUP BY a", "wildcard"),
            ("SELECT a FROM t WHERE c > 1 GROUP BY a", "1 item"),
            (
                "SELECT a, SUM(b), d FROM t WHERE c > 1 GROUP BY a",
                "3 item",
            ),
            (
                "SELECT SUM(b), a FROM t WHERE c > 1 GROUP BY a",
                "aggregate as the first",
            ),
            (
                "SELECT a, b FROM t WHERE c > 1 GROUP BY a",
                "plain column as the second",
            ),
            (
                "SELECT a, SUM(b + 1) FROM t WHERE c > 1 GROUP BY a",
                "expression inside SUM",
            ),
            ("SELECT a, SUM(*) FROM t WHERE c > 1 GROUP BY a", "SUM(*)"),
            (
                "SELECT t.a, SUM(b) FROM t WHERE c > 1 GROUP BY a",
                "qualified column",
            ),
        ];

        for (sql, expected) in cases {
            let what = rejection(sql);
            assert!(
                what.contains(expected),
                "for {sql:?}\n  expected message containing {expected:?}\n  got: {what}"
            );
        }
    }

    #[test]
    fn rejects_out_of_scope_predicates() {
        let cases = [
            ("SELECT a, SUM(b) FROM t WHERE c >= 1 GROUP BY a", ">="),
            ("SELECT a, SUM(b) FROM t WHERE c <= 1 GROUP BY a", "<="),
            ("SELECT a, SUM(b) FROM t WHERE c <> 1 GROUP BY a", "<>"),
            (
                "SELECT a, SUM(b) FROM t WHERE c > 1 AND d < 2 GROUP BY a",
                "AND / OR",
            ),
            (
                "SELECT a, SUM(b) FROM t WHERE c > 1 OR d < 2 GROUP BY a",
                "AND / OR",
            ),
            ("SELECT a, SUM(b) FROM t WHERE NOT c > 1 GROUP BY a", "NOT"),
            (
                "SELECT a, SUM(b) FROM t WHERE c BETWEEN 1 AND 2 GROUP BY a",
                "BETWEEN",
            ),
            ("SELECT a, SUM(b) FROM t WHERE c IN (1, 2) GROUP BY a", "IN"),
            (
                "SELECT a, SUM(b) FROM t WHERE c LIKE 'x%' GROUP BY a",
                "LIKE",
            ),
            (
                "SELECT a, SUM(b) FROM t WHERE c IS NULL GROUP BY a",
                "IS NULL",
            ),
            (
                "SELECT a, SUM(b) FROM t WHERE c > d GROUP BY a",
                "column-to-column",
            ),
            (
                "SELECT a, SUM(b) FROM t WHERE 1 < c GROUP BY a",
                "literal on the left",
            ),
        ];

        for (sql, expected) in cases {
            let what = rejection(sql);
            assert!(
                what.contains(expected),
                "for {sql:?}\n  expected message containing {expected:?}\n  got: {what}"
            );
        }
    }

    #[test]
    fn rejects_literal_types_the_engine_has_no_representation_for() {
        // Both tie back to documented scope cuts: no Bool column type (prd.md), and no NULL
        // support (established in Phase 1's loader).
        let boolean = rejection("SELECT a, SUM(b) FROM t WHERE c = TRUE GROUP BY a");
        assert!(boolean.contains("boolean"), "message was: {boolean}");

        let null = rejection("SELECT a, SUM(b) FROM t WHERE c = NULL GROUP BY a");
        assert!(null.contains("NULL"), "message was: {null}");
    }

    #[test]
    fn rejects_missing_mandatory_clauses() {
        let cases = [
            ("SELECT a, SUM(b) FROM t GROUP BY a", "no WHERE"),
            ("SELECT a, SUM(b) FROM t WHERE c > 1", "no GROUP BY"),
        ];

        for (sql, expected) in cases {
            let what = rejection(sql);
            assert!(
                what.contains(expected),
                "for {sql:?}\n  expected message containing {expected:?}\n  got: {what}"
            );
        }
    }

    #[test]
    fn rejects_more_than_one_statement() {
        let what = rejection(&format!("{SHAPE}; {SHAPE}"));
        assert!(what.contains("2 statements"), "message was: {what}");
    }
}
