from railrl.algos.bptt_ddpg import BpttDDPG


class MetaBpttDdpg(BpttDDPG):
    """
    Add a meta critic: it predicts the error of the normal critic
    """
    pass
