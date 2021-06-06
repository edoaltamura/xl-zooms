import literature

modules = dict(
    [(name, cls) for name, cls in literature.__dict__.items() if isinstance(cls, type)]
)

for module_name, module_class in modules.items():
    try:
        module_instance = module_class()
    except:
        pass