from prefect import task, flow

@task
def printer(obj):
    print(f"received a {type(obj)} with value {obj}")

@flow
def val_flow(x: int, y: str):
    printer(x)
    printer(y)

val_flow(x="22", y=55)