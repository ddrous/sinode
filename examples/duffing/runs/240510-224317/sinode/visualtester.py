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









    def visualize(self, data_loader, e=None, traj=None, dims=(0,1), int_cutoff=1.0, plot_tol=1e-6, save_path=False, key=None):

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

        losses_model = self.trainer.losses_model
        losses_coeffs = self.trainer.losses_coeffs

        ## Since loss can be negative
        min_loss = min(np.min(losses_model), np.min(losses_coeffs), 0.)
        losses_model += abs(min_loss) + plot_tol
        losses_coeffs += abs(min_loss) + plot_tol

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




    # def ablate_coeffs():
    #     """ Set some coefficients to zero and leave others unchanged, then plot """


    #     saved_gammas = coeffs_.gammas
    #     saved_lambdas = coeffs_.lambdas

    #     for i in range(1):
    #     # for i in range(dict_size):
    #     #     print("Setting zeros at position: ", i)

    #     #     new_lambdas = saved_lambdas.at[i].set(0.)
    #     #     new_gammas = saved_gammas.at[i].set(0.)

    #     #     # new_lambdas = saved_lambdas.at[i].mul(2.)
    #     #     # new_gammas = saved_gammas.at[i].mul(2.)

    #     #     # new_lambdas = jnp.zeros_like(saved_lambdas)
    #     #     # new_gammas = jnp.zeros_like(saved_gammas)
    #     #     # new_lambdas = new_lambdas.at[i].set(saved_lambdas[i])
    #     #     # new_gammas = new_gammas.at[i].set(saved_gammas[i])

    #     #     coeffs_ = eqx.tree_at(lambda m: m.lambdas, coeffs_, new_lambdas)
    #     #     coeffs_ = eqx.tree_at(lambda m: m.gammas, coeffs_, new_gammas)

    #         X_hat = test_model(model, coeffs_, (X[:, 0,:], t), test_key)

    #         print(f"Test MSE: {jnp.mean((X-X_hat)**2):.8f}")

    #         fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    #         colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'orange', 'yellow', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    #         colors = colors*10

    #         for i in range(X.shape[0]):
    #             if i==0:
    #                 sbplot(X_hat[i, :,0], X_hat[i, :,1], "-", x_label='x', y_label='y', label=f'Pred', title=f'Phase space', ax=ax, alpha=0.5, color=colors[i])
    #                 sbplot(X[i, :,0], X[i, :,1], "+", lw=1, label=f'True', ax=ax, color=colors[i])
    #             else:
    #                 sbplot(X_hat[i, :,0], X_hat[i, :,1], "-", x_label='x', y_label='y', ax=ax, alpha=0.5, color=colors[i])
    #                 sbplot(X[i, :,0], X[i, :,1], "+", lw=1, ax=ax, color=colors[i])

    #         ## Limit ax x and y axis to (-5,5)
    #         # ax.set_xlim(-5, 5)
    #         # ax.set_ylim(-5, 5)
    #         plt.show()


    #     plt.savefig(f"data/ablated_traj.png", dpi=300, bbox_inches='tight')











    # def symbolic_regression():
    #     # ## Print coeffs_
    #     print("coeffs_: \n\t  - Lambdas: \n", coeffs_.lambdas, "\n\n\t  - Gammas: \n", coeffs_.gammas)
    #     # print("coeffs_: \n\t  - Lambdas: \n", coeffs_.lambdas)

    #     ## Print cloeffs lambda with abs > 1e-2
    #     active_coeffs_ = jnp.where(jnp.abs(coeffs_.lambdas)>=threshold_val, 1, 0)
    #     print("Active coefficients lambda: \n", active_coeffs_)

    #     ## Same for gammas
    #     active_coeffs_ = jnp.where(jnp.abs(coeffs_.gammas)>=threshold_val, 1, 0)
    #     print("Active coefficients gammas: \n", active_coeffs_)

    #     ## Count the number of paramters in the model
    #     params = eqx.filter(model, eqx.is_array)
    #     nb_params = jnp.sum(jnp.array([jnp.prod(jnp.array(p.shape)) for p in jax.tree.flatten(params)[0]]))
    #     print(f"\nNumber of parameters in the model: {nb_params}")

    #     ## Print model basis functions
    #     # print("Model basis functions: ", model.vector_field.basis_funcs.layers)
    #     # print("Model basis functions: ", model.vector_field.basis_funcs.layers[2].weight)


    #     ## Evaluate the vector field a few points
    #     print("Vector field at [1,1]: \n", evaluate_funcs_dir(model.vector_field.basis_funcs, jnp.array([1., 1.])))
    #     print("Vector field at [2,2]: \n", evaluate_funcs_dir(model.vector_field.basis_funcs, jnp.array([2., 2.])))
