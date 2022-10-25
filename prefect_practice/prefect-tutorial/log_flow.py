from multiprocessing.util import get_logger
import sys
import prefect
from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner

from utilities import AN_IMPORTED_MESSAGE

@task
def log_task(name):
    logger = get_run_logger()
    logger.info("Hello %s!", name)
    return

@task    
def log_ver():
    logger = get_run_logger()
    logger.info("Prefect version = %s", prefect.__version__)

@task
def log_debug():
    logger = get_run_logger()
    logger.debug(AN_IMPORTED_MESSAGE)


@flow(task_runner=SequentialTaskRunner())
def log_flow(name: str):
    log_task.submit(name)
    log_ver()
    log_debug()

if __name__ == "__main__":
    name = "Mr. Data Scientist"
    log_flow(name)


