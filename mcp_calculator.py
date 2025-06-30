import math
import ast
import json
from loguru import logger
from operator import add, sub, mul, truediv, pow, mod

# Allowed mathematical functions and constants
_allowed_names = {
    "__builtins__": None, # Disable builtins
    "abs": abs,
    "round": round,
    # "min": min,
    # "max": max,
    # "sum": sum,
    # "len": len, # Safe for basic list/tuple operations
    "add": add,
    "sub": sub,
    "mul": mul,
    "div": truediv,
    "mod": mod,
    "math": {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "pi": math.pi,
        "e": math.e,
        "degrees": math.degrees,
        "radians": math.radians,
        "floor": math.floor,
        "ceil": math.ceil,
        "pow": math.pow,
        "fabs": math.fabs,
        "fmod": math.fmod,
        "gcd": math.gcd,
        "isclose": math.isclose,
        "isfinite": math.isfinite,
        "isinf": math.isinf,
        "isnan": math.isnan,
        "trunc": math.trunc,
    },
}

class CalculatorASTValidator(ast.NodeVisitor):
    def visit_Expression(self, node):
        self.visit(node.body)

    def visit_BinOp(self, node):
        self.visit(node.left)
        self.visit(node.op)
        self.visit(node.right)

    def visit_UnaryOp(self, node):
        self.visit(node.op)
        self.visit(node.operand)

    def visit_Call(self, node):
        # Check if the function being called is allowed
        if isinstance(node.func, ast.Name):
            if node.func.id not in _allowed_names:
                raise ValueError(f"Disallowed function call: {node.func.id}")
        elif isinstance(node.func, ast.Attribute):
            # Handle math.sqrt, etc.
            if not (isinstance(node.func.value, ast.Name) and node.func.value.id == "math" and node.func.attr in _allowed_names["math"]):
                raise ValueError(f"Disallowed attribute access or function call: {ast.dump(node.func)}")
        else:
            raise ValueError(f"Disallowed function call type: {type(node.func).__name__}")
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            raise ValueError("Keyword arguments are not allowed in calculator expressions.")

    def visit_Name(self, node):
        if node.id not in _allowed_names and node.id != "math": # 'math' itself is allowed as a module name
            raise ValueError(f"Disallowed name: {node.id}")

    def visit_Attribute(self, node):
        # Only allow access to attributes of 'math' module
        if not (isinstance(node.value, ast.Name) and node.value.id == "math" and node.attr in _allowed_names["math"]):
            raise ValueError(f"Disallowed attribute access: {ast.dump(node)}")

    def visit_Constant(self, node):
        # Allow numbers, True, False, None
        if not isinstance(node.value, (int, float, bool, type(None))):
            raise ValueError(f"Disallowed constant type: {type(node.value).__name__}")

    # Operators - all standard arithmetic operators are generally safe
    def visit_Add(self, node): pass
    def visit_Sub(self, node): pass
    def visit_Mult(self, node): pass
    def visit_Div(self, node): pass
    def visit_FloorDiv(self, node): pass
    def visit_Mod(self, node): pass
    def visit_Pow(self, node): pass
    def visit_USub(self, node): pass # Unary minus

    def generic_visit(self, node):
        # Raise an error for any unhandled node types (i.e., disallowed operations)
        raise ValueError(f"Disallowed AST node type: {type(node).__name__}")

def _evaluate_expression_ast(expression: str):
    try:
        # Parse the expression into an AST
        tree = ast.parse(expression, mode="eval")

        # Validate the AST
        validator = CalculatorASTValidator()
        validator.visit(tree)

        # Execute the safe AST in a restricted environment
        # We pass _allowed_names as both globals and locals to ensure only these are accessible
        return eval(compile(tree, "<ast>", "eval"), _allowed_names, {"math": _allowed_names["math"]})
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax in expression: {e}")
    except ValueError as e:
        raise e # Re-raise validation errors
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {e}")

async def calculate(name: str, arguments: dict) -> str:
    logger.debug(f"Calculating expression: {arguments.get('expression')}")
    if name != "calculate":
        raise ValueError(f"Unknown tool: {name}")
    expression = arguments.get("expression")
    if not expression:
        raise ValueError("Missing required argument 'expression' for calculate")
    try:
        result = _evaluate_expression_ast(expression)
        return json.dumps({"result": result})
    except (SyntaxError, TypeError, NameError, ValueError) as e:
        logger.error(f"Error calculating expression: {e}")
        return json.dumps({"error": f"Invalid expression or unsupported operation: {e}"})

CALCULATE_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "expression": {"type": "string", "description": "The mathematical expression to evaluate."}
    },
    "required": ["expression"]
}
