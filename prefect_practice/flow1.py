from prefect import flow

@flow
def my_func():
    print("Whats yot fav number??")
    return "22557799"



print(my_func())