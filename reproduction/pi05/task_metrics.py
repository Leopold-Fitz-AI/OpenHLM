"""Task scoring independent of rendering or a policy implementation."""
from dataclasses import dataclass, field
import math


def point_aabb_distance(point, low, high):
    """Distance to the whole bottle volume, including its neck and cap."""
    return math.sqrt(sum(max(lo-x, 0., x-hi)**2 for x,lo,hi in zip(point,low,high)))


@dataclass
class BottleBinScore:
    table_top: float
    bin_center: tuple[float, float]
    bin_half_size: tuple[float, float]
    bin_floor: float
    bin_rim: float
    bin_radius: float | None = None
    grasped: bool = False
    lifted: bool = False
    success: bool = False
    events: dict = field(default_factory=dict)
    _grip_since: float | None = None
    _lift_since: float | None = None
    _inside_since: float | None = None
    max_lift: float = 0.0

    def update(self, t, low, high, bilateral_contact, opening, close_command,
               linear_speed, angular_speed, near_gripper=True, any_gripper_contact=False):
        if not all(math.isfinite(x) for x in [t, *low, *high, opening, linear_speed, angular_speed]):
            raise ValueError("Non-finite task measurement")
        gripping = bilateral_contact and close_command < 0.8 and near_gripper
        self._grip_since = t if gripping and self._grip_since is None else self._grip_since if gripping else None
        if self._grip_since is not None and t - self._grip_since >= 0.15 and not self.grasped:
            self.grasped = True
            self.events["grasped"] = t
        clearance = low[2] - self.table_top
        self.max_lift = max(self.max_lift, clearance)
        lifting = self.grasped and gripping and clearance > 0.06
        self._lift_since = t if lifting and self._lift_since is None else self._lift_since if lifting else None
        if self._lift_since is not None and t - self._lift_since >= 0.25 and not self.lifted:
            self.lifted = True
            self.events["lifted"] = t
        # Contact solvers permit small penetration when the bottle rests against a wall.
        inside = all(low[i] >= self.bin_center[i] - self.bin_half_size[i] - 0.003 and
                     high[i] <= self.bin_center[i] + self.bin_half_size[i] + 0.003 for i in (0, 1))
        inside = inside and low[2] >= self.bin_floor - 0.015 and high[2] <= self.bin_rim - 0.01
        if self.bin_radius is not None:
            inside=inside and all(math.hypot(x-self.bin_center[0],y-self.bin_center[1])<=self.bin_radius+.003
                                  for x in [low[0],high[0]] for y in [low[1],high[1]])
        settled = self.lifted and inside and opening > 0.85 and close_command > 0.85
        settled = settled and not bilateral_contact and not any_gripper_contact and linear_speed < 0.15 and angular_speed < 1.0
        self._inside_since = t if settled and self._inside_since is None else self._inside_since if settled else None
        if self._inside_since is not None and t - self._inside_since >= 0.8 and not self.success:
            self.success = True
            self.events["released_and_settled_in_bin"] = t
        return inside

    def result(self):
        return {"grasped": self.grasped, "lifted": self.lifted, "success": self.success,
                "max_bottom_clearance_m": self.max_lift, "events_sim_time": self.events}
