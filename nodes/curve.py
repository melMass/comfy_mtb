import json


def deserialize_curve(curve):
    if isinstance(curve, str):
        curve = json.loads(curve)
    return curve


def serialize_curve(curve):
    if not isinstance(curve, str):
        curve = json.dumps(curve)
    return curve


class MTBCurve:
    """A basic FLOAT_CURVE input node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "curve": ("FLOAT_CURVE",),
            },
        }

    RETURN_TYPES = ("FLOAT_CURVE",)
    FUNCTION = "do_curve"

    CATEGORY = "mtb/curve"

    def do_curve(self, curve):
        return (curve,)


__nodes__ = [MTBCurve]
