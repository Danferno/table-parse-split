# Imports
from prettytable import PrettyTable

def list_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    table.align['Modules'] = 'l'
    table.align['Parameters'] = 'r'
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def describe_model(model):
    print(model)
    list_parameters(model=model)      