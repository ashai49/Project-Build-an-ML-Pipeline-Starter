import pytest
import pandas as pd
import wandb
import os


def pytest_addoption(parser):
    parser.addoption("--csv", action="store")
    parser.addoption("--ref", action="store")
    parser.addoption("--kl_threshold", action="store")
    parser.addoption("--min_price", action="store")
    parser.addoption("--max_price", action="store")


@pytest.fixture(scope='session')
def data(request):
    run = wandb.init(job_type="data_tests", resume=True)
    
    # Get the CSV path and clean it up
    csv_path = request.config.option.csv
    if csv_path.startswith('wandb-artifact://'):
        csv_path = csv_path.replace('wandb-artifact://', '')
    
    data_path = run.use_artifact(csv_path).file()
    if data_path is None:
        pytest.fail("You must provide the --csv option on the command line")

    df = pd.read_csv(data_path)
    return df


@pytest.fixture(scope='session')
def ref_data(request):
    os.environ['WANDB_TIMEOUT'] = '60'
    run = wandb.init(job_type="data_tests", resume=True)
    
    # Get the reference path and clean it up
    ref_path = request.config.option.ref
    if ref_path.startswith('wandb-artifact://'):
        ref_path = ref_path.replace('wandb-artifact://', '')
    
    data_path = run.use_artifact(ref_path).file()
    if data_path is None:
        pytest.fail("You must provide the --ref option on the command line")

    df = pd.read_csv(data_path)
    return df


@pytest.fixture(scope='session')
def kl_threshold(request):
    kl_threshold = request.config.option.kl_threshold
    if kl_threshold is None:
        pytest.fail("You must provide a threshold for the KL test")
    return float(kl_threshold)


@pytest.fixture(scope='session')
def min_price(request):
    min_price = request.config.option.min_price
    if min_price is None:
        pytest.fail("You must provide min_price")
    return float(min_price)


@pytest.fixture(scope='session')
def max_price(request):
    max_price = request.config.option.max_price
    if max_price is None:
        pytest.fail("You must provide max_price")
    return float(max_price)