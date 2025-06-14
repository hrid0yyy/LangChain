from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool
from langchain.tools import StructuredTool, BaseTool
from typing import Type
from pydantic import BaseModel, Field



search = DuckDuckGoSearchResults()
res = search.invoke("rcb ipl 2025?")

# Below one is a custom tool example with @tool decorator.
@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers."""
    return a * b

res = multiply.invoke({"a": 2, "b": 3})
print(multiply.name)
print(multiply.description)
print(multiply.args)
print(res)

# Structured tool example with pydantic model.
class AddInput(BaseModel):
    a: int = Field(required=True, description="First number to add")
    b: int = Field(required=True,description="Second number to add")

def add(a: int, b: int) -> int:
    return a + b

add_tool = StructuredTool.from_function(
    func=add,
    name="add",
    description="Adds two numbers.",
    args_schema=AddInput
)

res = add_tool.invoke({"a": 5, "b": 7})
print(add_tool.name)
print(add_tool.description)
print(add_tool.args)
print(res)


# Tools using BaseTool class. add two numbers.
class AddTool(BaseTool):
    name: str = "add"
    description: str = "Adds two numbers."

    args_schema: Type[BaseModel] = AddInput

    def _run(self, a: int, b: int) -> int:
        return a + b


add_tool_base = AddTool()
res = add_tool_base.invoke({"a": 7, "b": 7})
print(res)