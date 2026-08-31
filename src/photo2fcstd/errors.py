class Photo2FCStdError(Exception):
    pass


class CaptureError(Photo2FCStdError):
    pass


class BuildError(Photo2FCStdError):
    pass
