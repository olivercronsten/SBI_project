import sbibm
import numpy as np
import os 
import matplotlib.pyplot as plt
import bayesflow as bf
from bayesflow import benchmarks

def generate_dataset(task_name:str, 
                     num_samples:int,  
                     save_file=False,
                     custom_output_dir = None):
    
    """
    A function that generates an SBI dataset and either saves it to OS or returns it to a variable.

    Args:
        - task_name: The name of the model that the dataset should originate from. 
            Given an unknown model, the functions returns available models. 
        - num_samples: The number of samples that should be generated.
        - save_file: Controls whether the dataset should be saved to OS or assigned to a variable
            If True, no variable assignment needed.
        - custom_output_dir: If "save_file = True", a custom directory can be chosen, otherwise
            a "dataset"-folder will be created in the same directory as the function script.

    Returns:
        If save_file = True, return nothing. Otherwise, returns a dict.
    """

    #If save_fie, checking if dataset has already been created.
    #Otherwise, 
    if save_file:
        if custom_output_dir == None:
            custom_output_dir = os.path.dirname(os.path.abspath(__file__))

        #Defining where a folder of datasets should be
        datasets_path = os.path.join(custom_output_dir, "datasets")
        
        #Output path for specific task
        task_path = os.path.join(datasets_path, task_name)
        
        #Effectively naming the output file and the budget
        output_path = os.path.join(task_path, \
                f"training_data_{task_name}_budget_{int(num_samples/1000)}k.npz")
        
        #Do not regenerate already existing datasets.
        if not os.path.exists(output_path):
            
            #Checking if a datasets folder exists. If not, a subfolder in output_dir will be created
            if not os.path.exists(datasets_path):
                os.mkdir(datasets_path)

            #Checking if a folder for current task exists. If not, a subfolder to datasets will be created
            if not os.path.exists(task_path):
                os.mkdir(task_path)
        
        else:
            print(f"A dataset for {task_name} with {num_samples} samples already exists")
            return
    
    
    #If non-existing dataset, create one.
    try:
        if task_name not in ['lotka_volterra', 'sir']:
            task = sbibm.get_task(task_name)
            prior = task.get_prior()
            simulator = task.get_simulator()

            #Generating the training data and transforming to np arrays
            thetas_tensor = prior(num_samples)
            y_samples_tensor = simulator(thetas_tensor)
            np_thetas = thetas_tensor.numpy()
            np_y_samples = y_samples_tensor.numpy()

        else:
            def simulator(theta):
                x = benchmark.generative_model.simulator(theta)
                return x["sim_data"]

            def prior(N):
                x = benchmark.generative_model.prior(N)
                return x["prior_draws"]
        
            benchmark = benchmarks.Benchmark(task_name)
            np_thetas = prior(num_samples)
            np_y_samples = simulator(np_thetas)
        
        if not save_file:
            dataset = {'thetas': np_thetas,
                       'y_samples': np_y_samples}

            return dataset

        np.savez(output_path, thetas=np_thetas, y_samples=np_y_samples)
        print(f"A dataset for {task_name} with {num_samples} samples has been created!")

        
        
            
    #If unavailable task
    except NotImplementedError:
        print(f"Task name: [{task_name}] is not a defined task. See available tasks below: \n")
        print(sbibm.get_available_tasks())
        print("OR \n ['lotka_volterra', 'sir']")
        raise




