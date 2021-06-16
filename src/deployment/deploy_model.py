

from azureml.core.environment import Environment




if __name__ == "__main__":

    myenv = Environment.from_pip_requirements("mlops", file_path="requirements.txt")
