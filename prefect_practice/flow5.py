from prefect import flow

@flow
def com_flow(config: dict):
    print("I am a subgraph that shows up in lots of places!")
    intermediate_result = 22
    return intermediate_result

@flow
def main_flow(a:int, b:int):
    c = a + b
    data = com_flow(config={})
    return c

main_flow(2,3)
