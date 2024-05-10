#%%[markdown]
# # Neural ODE mixed with SINDy framework with mixture of experts

### Summary
# - We have a multistable dynamical systems that we want to learn
# - We only learn on a single attrator
# - Does it learn the other attractor ?

### Aproach
# - We have a baisis of neural networks
# - We have a set of coefficients: one set = one dataset
# - Hopefully we get a foundational model

#%%
%load_ext autoreload
%autoreload 2

import jax

print("Available devices:", jax.devices())

from jax import config
# config.update("jax_debug_nans", True)

import jax.numpy as jnp

import numpy as np
from scipy.integrate import solve_ivp

import equinox as eqx
import diffrax

# import matplotlib.pyplot as plt

from sinode import *

import optax
import time


#%%

## Main hps ##
SEED = 2026
main_key = jax.random.PRNGKey(SEED)
train = True

gen_data=True
run_folder="runs/240510-223414/"

## Data generation hps ##
T_horizon = 20
skip = 50


## Model hps ##
dict_size = 8
mlp_hidden_size = 32
mlp_depth = 4
threshold_val = 1e-2
threshold_every = 100    ## Threshold the coeffs every this many epochs
# renormalize_every = 100
use_bias = False
fixed_point_steps = 5


## Loss & optimizer hps ##
epsilon = 0e1  ## For contrastive loss
eta_inv, eta_cont, eta_spar = 1e-3, 1e-2, 1e-1

init_lr = 5e-4
sched_factor = 1.0

power_iter_steps = 5
spectral_scaling = 1.0 # since the spectral norm is under-estimated wia power iter


## (Proximal) Training hps ##
nb_outer_steps_max=2000
inner_tol_model=1e-9 
inner_tol_coeffs=1e-8
nb_inner_steps_max=10
proximal_beta=100.
patience=nb_outer_steps_max

## Other hps ##
print_every = 100


#%%
# Define the Duffing system
def duffing1(t, state, a, b, c):
    x, y = state
    dxdt = y
    dydt = a*y - x*(b + c*x**2)
    return [dxdt, dydt]

def duffing2(t, state, a, b, c):
    x, y = state
    dxdt = y
    dydt = -c*x**3 / 5.
    return [dxdt, dydt]

# Parameters
a, b, c = -1/2., -1, 1/10.

t_span = (0, T_horizon)
t_eval = np.arange(t_span[0], t_span[1], 0.01)[::skip]

init_conds_train = np.array([[-0.5, -1], [-0.5, -0.5], [-0.5, 0.5], 
                       [-1.5, 1], 
                    #    [-0.5, 1], 
                       [-1, -1], [-1, -0.5], [-1, 0.5], [-1, 1], 
                       [-2, -1], [-2, -0.5], [-2, 0.5], [-2, 1],
                    #    [0.5, -1], [0.5, -0.5], [0.5, 0.5], [0.5, 1],
                    #    [1, -1], [1, -0.5], [1, 0.5], [1, 1],
                    #    [2, -1], [2, -0.5], [2, 0.5], [2, 1],
                       ])

init_conds_test = np.array([[-0.5, -1], [-0.5, -0.5], [-0.5, 0.5], 
                    [-1.5, 1], 
                    #    [-0.5, 1], 
                    [-1, -1], [-1, -0.5], [-1, 0.5], [-1, 1], 
                    [-2, -1], [-2, -0.5], [-2, 0.5], [-2, 1],
                    [0.5, -1], [0.5, -0.5], [0.5, 0.5], [0.5, 1],
                    [1, -1], [1, -0.5], [1, 0.5], [1, 1],
                    [2, -1], [2, -0.5], [2, 0.5], [2, 1],
                    ])


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5*2), sharex=True)

if gen_data == True:

    test_data = np.zeros(shape=(2, len(init_conds_test), len(t_eval), 2))
    for i, state0 in enumerate(init_conds_test):
        sol1 = solve_ivp(duffing1, t_span, state0, args=(a, b, c), t_eval=t_eval)
        test_data[0, i] = sol1.y.T

        sol2 = solve_ivp(duffing2, t_span, state0, args=(a, b, c), t_eval=t_eval)
        test_data[1, i] = sol2.y.T

        ax1 = sbplot(sol1.y[0], sol1.y[1], ".-", color="grey", ax=ax1)
        ax2 = sbplot(sol2.y[0], sol2.y[1], ".-", color="grey", ax=ax2)

    train_data = np.zeros(shape=(2, len(init_conds_train), len(t_eval), 2))
    for i, state0 in enumerate(init_conds_train):
        sol1 = solve_ivp(duffing1, t_span, state0, args=(a, b, c), t_eval=t_eval)
        train_data[0, i] = sol1.y.T

        sol2 = solve_ivp(duffing2, t_span, state0, args=(a, b, c), t_eval=t_eval)
        train_data[1, i] = sol2.y.T

        ax1 = sbplot(sol1.y[0], sol1.y[1], ".-", ax=ax1)
        ax2 = sbplot(sol2.y[0], sol2.y[1], ".-", ax=ax2)


    ####----------------------------------------------------####
    run_folder = './runs/'+time.strftime("%y%m%d-%H%M%S")+'/'
    os.mkdir(run_folder)

    # Save the data
    np.savez(run_folder+"train_data.npz", X=train_data, t=t_eval)
    np.savez(run_folder+"test_data.npz", X=test_data, t=t_eval)

    # Save the run and dataset scripts in that folder
    script_name = os.path.basename(__file__)
    os.system(f"cp {script_name} {run_folder}")

    # Save the sinode module files as well
    os.system(f"cp -r ../../sinode {run_folder}")
    ####----------------------------------------------------####


else:
    ## Run folder must have been given up

    train_data, t_eval = np.load(run_folder+"train_data.npy").values()
    test_data, t_eval = np.load(run_folder+"test_data.npy").values()

    print("No training. Loading data from:", run_folder)

    for i in range(test_data.shape[1]):
        sol1 = test_data[0, i].T
        sol2 = test_data[1, i].T

        ax1 = sbplot(sol1[0], sol1[1], ".-", color="grey", ax=ax1)
        ax2 = sbplot(sol2[0], sol2[1], ".-", color="grey", ax=ax2)

    for i in range(train_data.shape[1]):
        sol1 = train_data[0, i].T
        sol2 = train_data[1, i].T

        ax1 = sbplot(sol1[0], sol1[1], ".-", ax=ax1)
        ax2 = sbplot(sol2[0], sol2[1], ".-", ax=ax2)

ax2.set_xlabel(r'$x$')
ax1.set_ylabel(r'$y$')
ax2.set_ylabel(r'$y$')
ax1.set_title('Env 1')
ax2.set_title('Env 2');


#%%

## Make the dataloaders
data_keys = jax.random.split(main_key)
train_dataloader = DataLoader(run_folder+"train_data.npz", batch_size=-1, shuffle=True, key=data_keys[0])
test_dataloader = DataLoader(run_folder+"test_data.npz", batch_size=-1, key=data_keys[1])


# %%

class BasisFunction(eqx.Module):
    """ An MLP with a skip connection from input to output, and its inverse via fixed point iteration """
    layers: jnp.ndarray

    # transform: jnp.ndarray  ## Of size 3: scale, shift, power

    def __init__(self, in_size, out_size, hidden_size, depth, activation, key=None):
        keys = jax.random.split(key, num=depth+1)

        self.layers = []

        for i in range(depth):
            if i==0:
                layer = eqx.nn.Linear(in_size, hidden_size, use_bias=use_bias, key=keys[i])
            elif i==depth-1:
                layer = eqx.nn.Linear(hidden_size, out_size, use_bias=use_bias, key=keys[i])
            else:
                layer = eqx.nn.Linear(hidden_size, hidden_size, use_bias=use_bias, key=keys[i])

            self.layers.append(layer)

            if i != depth-1:
                self.layers.append(activation)

        # self.transform = jnp.array([1.0, 0.0, 1.0, 0.0])   ## Identity transform


    def __call__(self, x):
        """ Returns y such that y = x + MLP(x) """
        y = x
        for layer in self.layers:
            y = layer(y)
        return x + y

    def inv_call(self, x):
        """ Returns z such that x = z + MLP(z) via fixed point iteration """
        # pass
        # return x

        z = x
        for _ in range(fixed_point_steps):
            z = x - self(z)
        return z

        # a, b, a_, b_ = self.transform
        # # return (a*self(x) + b)**p
        # return a*self(a_*x + b_) + b


## Vectorise across the model
@eqx.filter_vmap(in_axes=(eqx.if_array(0), None))
def evaluate_funcs_dir(model, x):
    return model(x)

@eqx.filter_vmap(in_axes=(eqx.if_array(0), None))
def evaluate_funcs_inv(model, x):
    return model.inv_call(x)


class VectorField(eqx.Module):
    basis_funcs: BasisFunction

    def __init__(self, data_size, dict_size, mlp_hidden_size, mlp_depth, key=None):
        keys = jax.random.split(key, num=dict_size)

        def make_basis_func(key):
            return BasisFunction(data_size, data_size, mlp_hidden_size, mlp_depth, jax.nn.softplus, key=key)

        self.basis_funcs = eqx.filter_vmap(make_basis_func)(keys)


    def __call__(self, t, x, coeffs):
        """ Forward call of the vector field """
        lambdas, gammas = coeffs
        # assert lambdas.shape == gammas.shape == (self.dict_size, self.data_size)

        y_direct = evaluate_funcs_dir(self.basis_funcs, x)
        y_inverse = evaluate_funcs_inv(self.basis_funcs, x)

        return jnp.sum(y_direct*lambdas + y_inverse*gammas, axis=0)
        # return jnp.array([y_direct[0,0], y_direct[1,1]])
        # return jnp.sum(y_direct*lambdas, axis=0)


class Coefficients(eqx.Module):
    lambdas: jnp.ndarray
    gammas: jnp.ndarray

    def __init__(self, data_size, dict_size, key=None):
        # self.lambdas = jax.random.uniform(key, shape=(dict_size, data_size))
        # self.gammas = jax.random.uniform(key, shape=(dict_size, data_size))

        self.lambdas = jnp.zeros((dict_size, data_size))

        # self.lambdas = self.lambdas.at[0, 0].set(1.)            ## TODO: Remove this
        # self.lambdas = self.lambdas.at[1, 1].set(1.)
        # self.lambdas = self.lambdas.at[2].set(1.)

        self.gammas = jnp.zeros((dict_size, data_size))

    def __call__(self):
        return self.lambdas, self.gammas

class NeuralODE(eqx.Module):
    data_size: int
    dict_size: int
    mlp_hidden_size: int
    mlp_depth: int

    vector_field: VectorField

    def __init__(self, data_size, dict_size, mlp_hidden_size, mlp_depth, key=None):
        self.data_size = data_size
        self.dict_size = dict_size
        self.mlp_hidden_size = mlp_hidden_size
        self.mlp_depth = mlp_depth

        self.vector_field = VectorField(data_size, dict_size, mlp_hidden_size, mlp_depth, key=key)

    def __call__(self, x0s, t_eval, coeffs):

        def integrate(y0):
            sol = diffrax.diffeqsolve(
                    diffrax.ODETerm(self.vector_field),
                    diffrax.Tsit5(),
                    args=(coeffs.lambdas, coeffs.gammas),
                    t0=t_eval[0],
                    t1=t_eval[-1],
                    dt0=1e-3,
                    y0=y0,
                    stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
                    saveat=diffrax.SaveAt(ts=t_eval),
                    # adjoint=diffrax.RecursiveCheckpointAdjoint(),
                    max_steps=4096
                )
            return sol.ys, sol.stats["n_steps"]

            # sol = RK4(self.vector_field, 
            #           (t_eval[0], t_eval[-1]), 
            #           y0, 
            #           (coeffs.lambdas, coeffs.gammas), 
            #           t_eval=t_eval, 
            #           subdivisions=4)
            # return sol, len(t_eval)*4

        trajs, nb_fes = eqx.filter_vmap(integrate)(x0s)
        return trajs, jnp.sum(nb_fes)


# %%

model_keys = jax.random.split(main_key, num=3)

model = NeuralODE(data_size=2, dict_size=dict_size, mlp_hidden_size=mlp_hidden_size, mlp_depth=mlp_depth, key=model_keys[0])

coeffs1 = Coefficients(data_size=2, dict_size=dict_size, key=model_keys[1])
coeffs2 = Coefficients(data_size=2, dict_size=dict_size, key=model_keys[2])
coeffs = (coeffs1, coeffs2)


# %%

def loss_rec(model, coeffs, batch, key):
    """ Reconstruction loss """
    X, t = batch
    X_hat, nb_fes = model(X[:, 0, :], t, coeffs)
    return jnp.mean((X-X_hat)**2), nb_fes

def loss_inv(model, coeffs, batch, key):
    """ Inverse loss - Assures the inverses are playing their roles """
    X, _ = batch

    @eqx.filter_vmap(in_axes=(None, 0))
    @eqx.filter_vmap(in_axes=(eqx.if_array(0), None))
    def eval_direct_inverse(basis_func, x):
        return basis_func(basis_func.inv_call(x))
    
    X_ = eval_direct_inverse(model.vector_field.basis_funcs, X.reshape(-1, model.data_size))
    X = X.reshape(-1, model.data_size)[:, None, :]

    return jnp.mean((X-X_)**2)

def loss_cont(model, coeffs, batch, key):
    """ Contrastive loss - Makes the weights of the models different """
    X, t = batch

    ## Extract and stack the weights along the last dimension
    basis_funcs = model.vector_field.basis_funcs
    weights = [layer.weight for layer in basis_funcs.layers if isinstance(layer, eqx.nn.Linear)]

    ## Sample two weights along the first dimention
    ind = jax.random.permutation(key, model.dict_size)[:2]
    # w1, w2 = weights[ind, ...]

    tot_distance = 0.
    for d in range(len(weights)):
        w1, w2 = weights[d][ind[0], ...], weights[d][ind[1], ...]
        tot_distance += jnp.mean((w1-w2)**2)

    return epsilon - tot_distance        ## Maximise the difference between the weights up to epsilon

def loss_sparsity(model, coeffs, batch, key):
    """ Sparsity loss - Makes the coefficients sparse """
    return jnp.mean(jnp.abs(coeffs.lambdas) + jnp.abs(coeffs.gammas))



def loss_fn(model, coeffs, batch, key):

    """ The loss considers one model, and also 1 coeffs (not a tuple)! """

    rec_loss, nb_fes = loss_rec(model, coeffs, batch, key)
    inv_loss = loss_inv(model, coeffs, batch, key)
    cont_loss = loss_cont(model, coeffs, batch, key)
    spar_loss = loss_sparsity(model, coeffs, batch, key)

    weighted_loss = rec_loss + eta_inv*inv_loss + eta_cont*cont_loss + eta_spar*spar_loss

    return weighted_loss, (rec_loss, inv_loss, cont_loss, spar_loss, nb_fes)



@eqx.filter_jit
def threshold_coeffs(coeffs, threshold_val):
    ## Jnp where every values above threshold is set to 1, else 0
    clip = lambda x: jnp.where(jnp.abs(x)>=threshold_val, x, 0.)
    return jax.tree.map(clip, coeffs)




@eqx.filter_jit
@eqx.filter_vmap(in_axes=(eqx.if_array(0), None, None))
def renormalize_model(model, nb_iters, key):
    """ Renormalize the model to have a spectral norm of 1 """

    def appply_func(x):
        if jnp.ndim(x)==2:
            spectral_norm = power_iteration(x, nb_iters, key)
            return jax.lax.cond((spectral_scaling / spectral_norm) < 1.,
                                lambda z: spectral_scaling * z / spectral_norm, 
                                lambda z: z,
                                operand=x)
        else:
            return x

    return jax.tree.map(appply_func, model)



#%%


total_steps = nb_outer_steps_max
boundaries_and_scales={int(total_steps*0.25):sched_factor, int(total_steps*0.5):sched_factor, int(total_steps*0.75):sched_factor}

sched_model = optax.piecewise_constant_schedule(init_value=init_lr, boundaries_and_scales=boundaries_and_scales)
sched_coeffs = optax.piecewise_constant_schedule(init_value=init_lr, boundaries_and_scales=boundaries_and_scales)

opt_model = optax.adam(sched_model)
opt_coeffs = (optax.adam(sched_coeffs), optax.adam(sched_coeffs))





""" Train the model using the proximal gradient descent algorithm. Algorithm 2 in https://proceedings.mlr.press/v97/li19n.html"""

train_key, _ = jax.random.split(main_key)

trainer = Trainer(model=model,
                  coeffs=coeffs,
                  loss_fn=loss_fn, 
                  optimizer_model=opt_model,
                  optimizer_coeffs=opt_coeffs,
                  )


if train == True:
    start_time = time.time()

    trainer.train_proximal(nb_outer_steps_max=nb_outer_steps_max, 
                            nb_inner_steps_max=nb_inner_steps_max, 
                            proximal_reg=proximal_beta, 
                            inner_tol_model=inner_tol_model, 
                            inner_tol_coeffs=inner_tol_coeffs,
                            print_error_every=print_every,
                            save_path=run_folder, 
                            val_dataloader=test_dataloader, 
                            patience=patience,
                            int_prop=1.0,
                            key=train_key)

    wall_time = time.time() - start_time
    time_in_hmsecs = seconds_to_hours(wall_time)
    print("\nTotal gradient descent training time: %d hours %d mins %d secs" %time_in_hmsecs)

else:
    print("\nNo training, loading model and results from "+ run_folder +" folder ...\n")

    trainer.restore_trainer(path=run_folder)





# %%


visualtester = VisualTester(trainer)
vis_key = jax.random.PRNGKey(time.time_ns())

visualtester.test(test_dataloader)

visualtester.visualize(test_dataloader, savefigdir=run_folder+"results.png", key=vis_key)





# %% [markdown]

# # Preliminary results
# - Nothing yet !

