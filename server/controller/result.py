class Result(dict):
    """
    结果对象
    """

    def __init__(self, isSuccess: bool, result: any = None, message: str = None):
        super().__init__()
        self["success"] = "success" if isSuccess else "fail"
        self["result"] = result
        self["message"] = message

    @classmethod
    def success(cls, result: any = None):
        return cls(isSuccess=True, result=result)

    @classmethod
    def fail(cls, message: str = None):
        return cls(isSuccess=False, message=message)
