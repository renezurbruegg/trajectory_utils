import roma
from roma.utils import quat_product, quat_conjugation

def unitquat_slerp(q0, q1, steps, shortest_arc=True):
    """
    Spherical linear interpolation between two unit quaternions.

    Args:
        q0, q1 (Ax4 tensor): batch of unit quaternions (A may contain multiple dimensions).
        steps (tensor of shape B): interpolation steps, 0.0 corresponding to q0 and 1.0 to q1 (B may contain multiple dimensions).
        shortest_arc (boolean): if True, interpolation will be performed along the shortest arc on SO(3) from `q0` to `q1` or `-q1`.
    Returns:
        batch of interpolated quaternions (BxAx4 tensor).
    Note:
        When considering quaternions as rotation representations,
        one should keep in mind that spherical interpolation is not necessarily performed along the shortest arc,
        depending on the sign of ``torch.sum(q0*q1,dim=-1)``.

        Behavior is undefined when using ``shortest_arc=False`` with antipodal quaternions.
    """
    # Relative rotation
    rel_q = quat_product(quat_conjugation(q0), q1)
    rel_rotvec = roma.mappings.unitquat_to_rotvec(rel_q, shortest_arc=shortest_arc)
    if steps.shape[0] != rel_rotvec.shape[0]:
        raise ValueError("The number of steps should be the same as the number of relative rotations.")

    # Relative rotations to apply
    rel_rotvecs = steps.reshape(steps.shape + (1,) * rel_rotvec.dim()) * rel_rotvec.reshape((1,) * steps.dim() + rel_rotvec.shape)
    rel_rotvecs = steps.unsqueeze(1) * rel_rotvec

    rots = roma.mappings.rotvec_to_unitquat(rel_rotvecs.reshape(-1, 3)).reshape(*rel_rotvecs.shape[:-1], 4)
    interpolated_q = quat_product(q0, rots)

    return interpolated_q