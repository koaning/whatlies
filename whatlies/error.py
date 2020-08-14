class NotInstalled:
    def __init__(self, tool, dep):
        self.tool = tool
        self.dep = dep

    def __call__(self, *args, **kwargs):
        msg = f"In order to use {self.tool} you'll need to install via;\n\n"
        msg += f"pip install whatlies[{self.dep}]\n\n"
        msg += "See installation guide here: https://rasahq.github.io/whatlies/#installation."
        raise ModuleNotFoundError(msg)
