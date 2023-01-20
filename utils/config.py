class Config:
    
    def __init__(self, **args): 
        for key, value in args.items():
            if type(value) == dict:
                args[key] = Config(**value)
    
        self.__dict__.update(args)