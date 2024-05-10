import warnings
from ._utils import *


class VisualTester:
    def __init__(self, trainer, *, key=None):
        self.key = key
        self.trainer = trainer


    def test(self, data_loader, criterion=None, int_cutoff=1.0, verbose=True):
        """ Compute test metrics on the adaptation dataloader  """

        criterion = criterion if criterion else lambda x, x_hat: jnp.mean((x-x_hat)**2)

        t_eval = data_loader.t_eval
        test_length = int(data_loader.nb_steps_per_traj*int_cutoff)
        X = data_loader.dataset[:, :, :test_length, :]
        t_test = t_eval[:test_length]

        if verbose == True:
            if data_loader.adaptation == False:
                print("==  Begining in-domain testing ... ==")
                print("    Number of training environments:", self.trainer.dataloader.nb_envs)
            else:
                print("==  Begining out-of-distribution testing ... ==")
                print("    Number of training environments:", self.trainer.dataloader.nb_envs)
                print("    Number of adaptation environments:", data_loader.nb_envs)
            print("    Final length of the training trajectories:", self.trainer.dataloader.int_cutoff)
            print("    Length of the testing trajectories:", test_length)

        assert data_loader.nb_envs == len(self.trainer.coeffs), "The number of environments in the test dataloader must be the same as the number coefficients."

        X_hat = np.zeros_like(X)
        for e in range(data_loader.nb_envs):
            X_hat[e, ...], _ = jax.vmap(self.trainer.model, in_axes=(0, None, 0))(
                    X[e, :, 0, :], 
                    t_test, 
                    self.trainer.coeffs[e]
                )

        batched_criterion = jax.vmap(jax.vmap(criterion, in_axes=(0, 0)), in_axes=(0, 0))

        crit_all = batched_criterion(X, X_hat).mean(axis=1)
        crit = crit_all.mean(axis=0)

        if verbose == True:
            if data_loader.adaptation == False:
                print("Test Score (In-Domain):", crit)
            else:
                print("Test Score (OOD):", crit)
            print(flush=True)

        return crit, crit_all









    def visualize(self, data_loader, e=None, traj=None, dims=(0,1), int_cutoff=1.0, save_path=False, key=None):

        if e is None or traj is None:
            if key is None:
                ValueError("You must provide a key if no environment or trajectory id was provided.")
            else:
                e_key, traj_key = jax.random.split(key)

        e = e if e is not None else jax.random.randint(e_key, (1,), 0, data_loader.nb_envs)[0]
        traj = traj is not None if traj else jax.random.randint(traj_key, (1,), 0, data_loader.nb_trajs_per_env)[0]

        t_eval = data_loader.t_eval
        test_length = int(data_loader.nb_steps_per_traj*int_cutoff)
        X = data_loader.dataset[e, traj:traj+1, :test_length, :]
        t_test = t_eval[:test_length]

        if data_loader.adaptation == False:
            print("==  Begining in-domain visualisation ... ==")
        else:
            print("==  Begining out-of-distribution visualisation ... ==")
        print("    Environment id:", e)
        print("    Trajectory id:", traj)
        print("    Visualized dimensions:", dims)
        print("    Final length of the training trajectories:", self.trainer.dataloader.int_cutoff)
        print("    Length of the testing trajectories:", test_length)

        # if data_loader.adaptation == False:
        #     contexts = self.trainer.learner.contexts.params
        # else:
        #     contexts = self.trainer.learner.contexts_adapt.params

        X_hat, _ = self.trainer.model(X[0, :], t_test, self.trainer.coeffs[e])

        fig, ax = plt.subplot_mosaic('AB;CC', figsize=(6*2, 3.5*2))

        mks = 2
        dim0, dim1 = dims

        ax['A'].plot(t_test, X[:, dim0], "o", c="deepskyblue", label=f"$x_{{{dim0}}}$ (True)")
        ax['A'].plot(t_test, X_hat[:, dim0], c="royalblue", label=f"$\\hat{{x}}_{{{dim0}}}$ (Pred)", markersize=mks)

        ax['A'].plot(t_test, X[:, dim1], "x", c="violet", label=f"$x_{{{dim1}}}$ (True)")
        ax['A'].plot(t_test, X_hat[:, dim1], c="purple", label=f"$\\hat{{x}}_{{{dim1}}}$ (Pref)", markersize=mks)

        ax['A'].set_xlabel("Time")
        ax['A'].set_ylabel("State")
        ax['A'].set_title("Trajectories")
        ax['A'].legend()

        ax['B'].plot(X[:, dim0], X[:, dim1], c="turquoise", label="True")
        ax['B'].plot(X_hat[:, dim0], X_hat[:, dim1], ".", c="teal", label="Pred")
        ax['B'].set_xlabel(f"$x_{{{dim0}}}$")
        ax['B'].set_ylabel(f"$x_{{{dim1}}}$")
        ax['B'].set_title("Phase space")
        ax['B'].legend()

        losses_model = np.vstack(self.trainer.losses_model)
        losses_coeffs = np.vstack(self.trainer.losses_coeffs)

        mke = np.ceil(losses_model.shape[0]/100).astype(int)

        label_model = "Model Loss" if data_loader.adaptation == False else "Node Loss Adapt"
        ax['C'].plot(losses_model[:], label=label_model, color="grey", linewidth=3, alpha=1.0)

        label_coeffs = "Coeffs Loss" if data_loader.adaptation == False else "Context Loss Adapt"
        ax['C'].plot(losses_coeffs[:], "x-", markevery=mke, markersize=mks, label=label_coeffs, color="grey", linewidth=1, alpha=0.5)

        if data_loader.adaptation==False and len(self.trainer.val_losses)>0:
            val_losses = np.concatenate(self.trainer.val_losses)
            ax['C'].plot(val_losses[:,0], val_losses[:,1], "y.", label="Validation Loss", linewidth=3, alpha=0.5)

        ax['C'].set_xlabel("Epochs")
        ax['C'].set_title("Loss Terms")
        ax['C'].set_yscale('log')
        ax['C'].legend()

        plt.suptitle(f"Results for env={e}, traj={traj}", fontsize=14)

        plt.tight_layout()
        plt.draw();

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print("Testing finished. Figure saved in:", save_path);




    def ablate_coeffs():
        """ Set some coefficients to zero and leave others unchanged, then plot """
        pass

