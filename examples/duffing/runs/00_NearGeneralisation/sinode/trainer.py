import pickle
import os

from sinode.dataloader import DataLoader
from sinode.visualtester import VisualTester
from ._utils import *



class Trainer:
    def __init__(self, model, coeffs, loss_fn, optimizer_model, optimizer_coeffs):

        self.model = model
        self.coeffs = coeffs

        self.loss_fn = loss_fn

        self.opt_model = optimizer_model
        self.opt_coeffs = optimizer_coeffs

        self.opt_model_state = self.opt_model.init(eqx.filter(self.model, eqx.is_array))
        self.opt_coeffs_state = self.opt_coeffs.init(self.coeffs)

        self.losses_model = None
        self.losses_coeffs = None

        self.val_losses = []




    def train_proximal(self, 
                       train_dataloader, 
                       nb_outer_steps_max=10, 
                       int_prop=1.0, 
                       inner_tol_model=1e-2, 
                       inner_tol_coeffs=1e-2, 
                       nb_inner_steps_max=10, 
                       proximal_reg=100., 
                       patience=None, 
                       print_every=1, 
                       threshold_val=1e-2,          ## Threshold value
                       threshold_every=100,         ## Threshold the coefficients every threshold_every steps
                       save_path=False, 
                       val_dataloader=None, 
                       val_criterion=None, 
                       *, key):
        """ Algorithm 2 in https://proceedings.mlr.press/v97/li19n.html 
        """

        ## Assert nb_envs == len(coeffs)
        assert train_dataloader.nb_envs == len(self.coeffs), "ERROR: Number of environments in the dataloader must match the number of coefficients provided."


        key = key

        opt_model = self.opt_model
        opt_coeffs = self.opt_coeffs

        opt_state_model = self.opt_model_state
        opt_state_coeffs = self.opt_coeffs_state

        loss_fn = self.loss_fn

        model = self.model
        coeffs = self.coeffs

        if val_dataloader is not None:
            tester = VisualTester(self)

        @eqx.filter_jit
        def train_step_model(model, model_old, coeffs, batch, opt_state, key):
            print('\nCompiling function "train_step" for neural ode ...')

            nb_envs = len(coeffs)

            def prox_loss_fn(model, coeffs, batch, key):
                loss, aux_data = loss_fn(model, coeffs, batch, key)
                diff_norm = params_diff_norm_squared(model, model_old)
                return loss + proximal_reg*diff_norm/2., (*aux_data, diff_norm)

            grad_loss_fn = eqx.filter_value_and_grad(prox_loss_fn, has_aux=True)

            loss = []
            aux_data = []
            for e in range(nb_envs):
                (loss_, aux_data_), grads_ = grad_loss_fn(model, coeffs[e], (batch[0][e], batch[1]), key)

                updates, opt_state = opt_model.update(grads_, opt_state)
                model = eqx.apply_updates(model, updates)

                loss.append(loss_)
                aux_data.append(aux_data_)

            ##---- Compute the average quantities to return ----##
            avg_loss = loss[0]
            avg_aux_data = list(aux_data[0])
            len_aux_data = len(avg_aux_data)

            for e in range(1, nb_envs):
                avg_loss += loss[e]

                aux_data_ = aux_data[e]
                for i in range(len_aux_data):
                    avg_aux_data[i] += aux_data_[i]

            avg_loss /= nb_envs
            for i in range(len_aux_data):
                avg_aux_data[i] /= nb_envs
            ##--------------------------------##

            return model, coeffs, opt_state, avg_loss, tuple(avg_aux_data)


        @eqx.filter_jit
        def train_step_coeffs(model, coeffs, coeffs_old, batch, opt_state, key):
            print('\nCompiling function "train_step" for coeffs ...')

            nb_envs = len(coeffs)

            def prox_loss_fn(coeffs, model, batch, coeffs_old, key):
                loss, aux_data = loss_fn(model, coeffs, batch, key)
                diff_norm = params_diff_norm_squared(coeffs, coeffs_old)
                return loss + proximal_reg * diff_norm / 2., (*aux_data, diff_norm)

            grad_loss_fn = eqx.filter_value_and_grad(prox_loss_fn, has_aux=True)

            loss = []
            aux_data = []
            all_grads =[]

            for e in range(nb_envs):
                (loss_, aux_data_), grads_ = grad_loss_fn(coeffs[e], model, (batch[0][e], batch[1]), coeffs_old[e], key)

                all_grads.append(grads_)

                loss.append(loss_)
                aux_data.append(aux_data_)

            updates, opt_state = opt_coeffs.update(tuple(all_grads), opt_state)
            coeffs = eqx.apply_updates(coeffs, updates)

            ##---- Compute the average loss to return ----##
            avg_loss = loss[0]
            for e in range(1, nb_envs):
                avg_loss += loss[e]
            avg_loss /= nb_envs
            ##--------------------------------##

            return model, coeffs, opt_state, avg_loss, (loss, tuple(aux_data))


        @eqx.filter_jit
        def threshold_coeffs(coeffs, threshold_val):
            ## Jnp where every values above threshold is set to 1, else 0
            clip = lambda x: jnp.where(jnp.abs(x)>=threshold_val, x, 0.)
            return jax.tree.map(clip, coeffs)


        print(f"\n\n=== Beginning training with proximal alternating minimization ... ===")
        print(f"    Number of examples in a batch: {train_dataloader.batch_size}")
        print(f"    Maximum number of steps per inner minimization: {nb_inner_steps_max}")
        print(f"    Maximum number of outer minimizations: {nb_outer_steps_max}")
        print(f"    Maximum total number of training steps: {nb_outer_steps_max*nb_inner_steps_max}")

        start_time = time.time()

        losses_model = []
        losses_coeffs = []

        train_key, _ = jax.random.split(key)

        early_stopping_count = 0

        for out_step in range(nb_outer_steps_max):

            model_old = model
            coeffs_old = tuple([coeffs_ for coeffs_ in coeffs])

            model_prev = model
            for in_step_model in range(nb_inner_steps_max):

                nb_batches_model = 0
                loss_sum_model = 0.
                
                for _, batch in enumerate(train_dataloader):

                    train_key, _ = jax.random.split(train_key)

                    model, coeffs, opt_state_model, loss_model, aux_data_model = train_step_model(model, model_old, coeffs, batch, opt_state_model, train_key)

                    loss_sum_model += loss_model
                    nb_batches_model += 1

                diff_model = params_diff_norm_squared(model, model_prev) / params_norm_squared(model_prev)
                if diff_model < inner_tol_model or out_step==0:
                    break
                model_prev = model

            loss_epoch_model = loss_sum_model/nb_batches_model


            coeffs_prev = coeffs
            for in_step_coeffs in range(nb_inner_steps_max):

                nb_batches_ctx = 0
                loss_sum_coeffs = 0.

                for _, batch in enumerate(train_dataloader):

                    train_key, _ = jax.random.split(train_key)

                    model, coeffs, opt_state_coeffs, loss_coeffs, aux_data_coeffs = train_step_coeffs(model, coeffs, coeffs_old, batch, opt_state_coeffs, train_key)

                    loss_sum_coeffs += loss_coeffs
                    nb_batches_ctx += 1

                diff_coeffs = params_diff_norm_squared(coeffs, coeffs_prev) / params_norm_squared(coeffs_prev)
                if diff_coeffs < inner_tol_coeffs or out_step==0:
                    break
                coeffs_prev = coeffs

            loss_epoch_coeffs = loss_sum_coeffs/nb_batches_ctx

            if (threshold_val is not None) and (out_step%threshold_every)==0:
                coeffs = threshold_coeffs(coeffs, threshold_val)

            losses_model.append(loss_epoch_model)
            losses_coeffs.append(loss_epoch_coeffs)

            if out_step%print_every==0 or out_step==nb_outer_steps_max-1:

                self.losses_model = jnp.array(losses_model)
                self.losses_coeffs = jnp.array(losses_coeffs)

                self.opt_model = opt_model
                self.opt_coeffs = opt_coeffs

                self.opt_model_state = opt_state_model
                self.opt_coeffs_state = opt_state_coeffs

                self.model = model
                self.coeffs = coeffs

                if val_dataloader is not None:

                    ind_crit,_ = tester.test(val_dataloader, int_cutoff=1.0, criterion=val_criterion, verbose=False)
                    self.val_losses.append(np.array([out_step, ind_crit])[None])

                    print(f"    Outer Step: {out_step:-5d}      Loss: {loss_epoch_model:-.8f}     ValIndCrit: {ind_crit:-.8f}", flush=True)

                    ## Check if val loss is lowest to save the model
                    if ind_crit <= jnp.concatenate(self.val_losses)[:, 1].min() and save_path:
                        print(f"        Saving best model so far ...")
                        self.save_trainer(save_path)

                        ## Save aux_data 
                        self.save_auxillary_data(save_path, aux_data_model, aux_data_coeffs, out_step)

                    ## Restore the learner at the last evaluation step
                    if out_step == nb_outer_steps_max-1:
                        print(f"        Setting the model to the best one found ...")
                        self.restore_trainer(save_path, include_losses=False)

                else:
                    print(f"    Outer Step: {out_step:-5d}      Loss: {loss_epoch_model:-.8f}", flush=True)

                print(f"        -NbInnerStepsModel: {in_step_model+1:4d}\n        -NbInnerStepsCoeffs: {in_step_coeffs+1:4d}\n        -DiffModel: {diff_model:.2e}\n        -DiffCoeffs:  {diff_coeffs:.2e}", flush=True)

            if in_step_model < 1 and in_step_coeffs < 1:
                early_stopping_count += 1
            else:
                early_stopping_count = 0

            if (patience is not None) and (early_stopping_count >= patience):
                print(f"Stopping early after {patience} steps with no improvement in the loss.")
                break


        wall_time = time.time() - start_time
        time_in_hmsecs = seconds_to_hours(wall_time)
        print("\nTraining time: %d hours %d mins %d secs" %time_in_hmsecs)


        # Save the model and results
        if save_path:
            self.save_trainer(save_path)



    def save_trainer(self, path):
        assert path[-1] == "/", "ERROR: The path must end with /"
        # print(f"\nSaving model and results into {path} folder ...\n")

        if not os.path.exists(path+"artifacts/"):
            os.makedirs(path+"artifacts/")

        eqx.tree_serialise_leaves(path+"model.eqx", self.model)
        eqx.tree_serialise_leaves(path+"coeffs.eqx", self.coeffs)

        path_ar = path+"artifacts/"
        pickle.dump(self.opt_model_state, open(path_ar+"opt_state_model.pkl", "wb"))
        pickle.dump(self.opt_coeffs_state, open(path_ar+"opt_state_coeffs.pkl", "wb"))

        np.save(path_ar+"train_loss_model.npy", self.losses_model)
        np.save(path_ar+"train_loss_coeffs.npy", self.losses_coeffs)
        np.save(path_ar+"val_loss_model.npy", np.concatenate(self.val_losses))


    def restore_trainer(self, path, include_losses=True):
        assert path[-1] == "/", "ERROR: Invalidn provided. The path must end with /"
        # print(f"\nLoading model and results from {path} folder ...\n")

        self.model = eqx.tree_deserialise_leaves(path+"model.eqx", self.model)
        self.coeffs = eqx.tree_deserialise_leaves(path+"coeffs.eqx", self.coeffs)

        path_ar = path+"artifacts/"
        self.opt_state_model = pickle.load(open(path_ar+"opt_state_model.pkl", "rb"))
        self.opt_state_coeffs = pickle.load(open(path_ar+"opt_state_coeffs.pkl", "rb"))

        if include_losses:
            self.losses_model = np.load(path_ar+"train_loss_model.npy")
            self.losses_coeffs = np.load(path_ar+"train_loss_coeffs.npy")
            self.val_losses = [np.load(path_ar+"val_loss_model.npy")]


    def save_auxillary_data(self, path, aux_data_model, aux_data_coeffs, step):
        assert path[-1] == "/", "ERROR: The path must end with /"
        # print(f"\nSaving auxillary data into {path} folder ...\n")

        if not os.path.exists(path+"artifacts/"):
            os.makedirs(path+"artifacts/")
        if not os.path.exists(path+"artifacts/aux_model/"):
            os.makedirs(path+"artifacts/aux_model/")
        if not os.path.exists(path+"artifacts/aux_coeffs/"):
            os.makedirs(path+"artifacts/aux_coeffs/")

        pickle.dump(aux_data_model, open(f"{path}artifacts/aux_model/{step:04d}.pkl", "wb"))
        pickle.dump(aux_data_coeffs, open(f"{path}artifacts/aux_coeffs/{step:04d}.pkl", "wb"))
