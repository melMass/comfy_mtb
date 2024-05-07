import json

from ..log import log


def deserialize_curve(curve):
    if isinstance(curve, str):
        curve = json.loads(curve)
    return curve


def serialize_curve(curve):
    if not isinstance(curve, str):
        curve = json.dumps(curve)
    return curve


class MTB_Curve:
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
        log.debug(f"Curve: {curve}")
        return (curve,)


class MTB_CurveToFloat:
    """Convert a FLOAT_CURVE to a FLOAT or FLOATS"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "curve": ("FLOAT_CURVE", {"forceInput": True}),
                "steps": ("INT", {"default": 10, "min": 2}),
            },
        }

    RETURN_TYPES = ("FLOATS", "FLOAT")
    FUNCTION = "do_curve"

    CATEGORY = "mtb/curve"

    def do_curve(self, curve, steps):
        log.debug(f"Curve: {curve}")

        # sort by x (should be handled by the widget)
        sorted_points = sorted(curve.items(), key=lambda item: item[1]["x"])
        # Extract X and Y values
        x_values = [point[1]["x"] for point in sorted_points]
        y_values = [point[1]["y"] for point in sorted_points]
        # Calculate step size
        step_size = (max(x_values) - min(x_values)) / (steps - 1)

        # Interpolate Y values for each step
        interpolated_y_values = []
        for step in range(steps):
            current_x = min(x_values) + step_size * step

            # Find the indices of the two points between which the current_x falls
            idx1 = max(idx for idx, x in enumerate(x_values) if x <= current_x)
            idx2 = min(idx for idx, x in enumerate(x_values) if x >= current_x)

            # If the current_x matches one of the points, no interpolation is needed
            if current_x == x_values[idx1]:
                interpolated_y_values.append(y_values[idx1])
            elif current_x == x_values[idx2]:
                interpolated_y_values.append(y_values[idx2])
            else:
                # Interpolate Y value using linear interpolation
                y1 = y_values[idx1]
                y2 = y_values[idx2]
                x1 = x_values[idx1]
                x2 = x_values[idx2]
                interpolated_y = y1 + (y2 - y1) * (current_x - x1) / (x2 - x1)
                interpolated_y_values.append(interpolated_y)

        return (interpolated_y_values, interpolated_y_values)


__nodes__ = [MTB_Curve, MTB_CurveToFloat]
