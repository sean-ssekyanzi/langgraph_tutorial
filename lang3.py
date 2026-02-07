from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool



tavily_tool = TavilySearchResults(max_results=3)

@tool
def multiply(a:int,b:int)->int:
    """this tool will do multiplication"""
    return a*b

@tool
def add(a:int, b:int) -> int:
    "Add two integers"
    return a+b

tools_list = [tavily_tool, multiply, add]


llm_with_tool = ChatOpenAI(model='gpt-4o-mini').bind_tools(tools_list)


class State(TypedDict):
    messages : Annotated[list[AnyMessage], add_messages]

#Define a node for the LLM chatbot
def llm_chatbot(state:State):
    # invoke the LLM with the current message history 
    return {'messages':[llm_with_tool.invoke(state['messages'])]}

#Toolnode will run the tools requested by the last AI message
#if there multiple tools called, it will run in parallel
tool_node = ToolNode(tools_list)#Accepts a list of tools

#Build the StateGraph
build = StateGraph(State)
build.add_node('LLM', llm_chatbot)
build.add_node('tools', tool_node)#Node to execute tools 

build.add_edge(START, 'LLM')#Start by sending user input to the LLM

#add conditional edge from 'LLM'
build.add_conditional_edges("LLM",
                            tools_condition,)

build.add_edge('tools','LLM')

app = build.compile()




