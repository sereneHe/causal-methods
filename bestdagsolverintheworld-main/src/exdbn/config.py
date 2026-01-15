class MilpConfig:
    def __init__(self, param1, param2, time_limit=18000, lambda1=0.1, lambda2=0.1, loss_type='default', constraints_mode='default', callback_mode='default', robust=False):
        self.param1 = param1
        self.param2 = param2
        self.time_limit = time_limit
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.loss_type = loss_type
        self.constraints_mode = constraints_mode
        self.callback_mode = callback_mode
        self.robust = robust
        self.weights_bound = 1.0  # Updated weights_bound to have a default numeric value
        self.reg_type = 'l2'  # Added default value for reg_type
        self.a_reg_type = 'l2'  # Added default value for a_reg_type
        self.target_mip_gap = 0.01  # Set default value for target_mip_gap


def make_milp_cfg(cfg: MilpConfig):
    # Return the MilpConfig object directly
    return cfg