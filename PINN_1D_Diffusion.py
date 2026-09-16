import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from keras import backend as K 
import matplotlib.font_manager as font_manager
font = font_manager.FontProperties(family= 'serif',
        weight= 'normal',
        size= 14,
)
fontdic = {'family': 'serif',
        'weight': 'normal',
        'color': 'darkred', 
        'size': '14',}
# -------------------------------
# Define the PINN neural network
# -------------------------------
class PINN(tf.keras.Model):
    def __init__(self, layers):
        """
        layers: list specifying the number of neurons in each layer.
                For example: [2, 50, 50, 50, 1] means 2 inputs (x,t),
                three hidden layers with 50 neurons each, and 1 output.
        """
        super(PINN, self).__init__()
        self.hidden_layers = []
        for width in layers[1:-1]:
            self.hidden_layers.append(tf.keras.layers.Dense(width, activation=tf.nn.tanh))
        self.out_layer = tf.keras.layers.Dense(layers[-1], activation=None)
    
    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        return self.out_layer(x)

def reinitialize(model):
    for l in model.layers:
        if hasattr(l,"kernel_initializer"):
            l.kernel.assign(l.kernel_initializer(tf.shape(l.kernel)))
        if hasattr(l,"bias_initializer"):
            l.bias.assign(l.bias_initializer(tf.shape(l.bias)))
        if hasattr(l,"recurrent_initializer"):
            l.recurrent_kernel.assign(l.recurrent_initializer(tf.shape(l.recurrent_kernel)))

# Define the neural network architecture
#layers= [2, 32, 32, 32, 1]  # 2 inputs: x and t, and 1 output: u(x,t)
"""
model = PINN(layers)
"""
# -------------------------------
# Define the PDE residual function
# -------------------------------
def pde_residual(x, t):
    """
    Computes the residual of the PDE:
        u_t - u_xx - cos(x) = 0.
    """
    # Use nested GradientTapes for second derivatives
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(x)
        tape2.watch(t)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(x)
            tape1.watch(t)
            X = tf.concat([x, t], axis=1)  # shape (N,2)
            u = model(X)
        u_x = tape1.gradient(u, x)
        u_t = tape1.gradient(u, t)
    u_xx = tape2.gradient(u_x, x)
    del tape1, tape2

    # Forcing term: f(x) = cos(x)
    f = tf.cos(x)
    # The PDE is u_t - u_xx = cos(x) or equivalently:
    #   u_t - u_xx - cos(x) = 0.
    return u_t - u_xx - f

# -------------------------------
# Define the loss function
# -------------------------------
def compute_loss(x_int, t_int, x_ic, t_ic, u_ic, x_lb, t_lb, u_lb, x_ub, t_ub, u_ub):
    """
    Computes the total loss as the sum of:
      (i) the mean square error (MSE) of the PDE residual
      (ii) the MSE of the initial condition
      (iii) the MSE of the boundary conditions.
    """
    # (i) PDE residual loss on collocation (interior) points
    f_pred = pde_residual(x_int, t_int)
    mse_pde = tf.reduce_mean(tf.square(f_pred))
    
    # (ii) Initial condition: u(x,0)=0
    X_ic = tf.concat([x_ic, t_ic], axis=1)
    u_ic_pred = model(X_ic)
    mse_ic = tf.reduce_mean(tf.square(u_ic_pred - u_ic))
    
    # (iii) Boundary conditions:
    # Left boundary: u(0,t)=1-exp(-t)
    X_lb = tf.concat([x_lb, t_lb], axis=1)
    u_lb_pred = model(X_lb)
    # Right boundary: u(pi,t)=-1+exp(-t)
    X_ub = tf.concat([x_ub, t_ub], axis=1)
    u_ub_pred = model(X_ub)
    mse_bc = tf.reduce_mean(tf.square(u_lb_pred - u_lb)) + tf.reduce_mean(tf.square(u_ub_pred - u_ub))
    
    return mse_pde + 1*mse_ic + 5*mse_bc



@tf.function
def train_step():
    with tf.GradientTape() as tape:
        loss_value = compute_loss(x_int_tf, t_int_tf,
                                  x_ic_tf, t_ic_tf, u_ic_tf,
                                  x_lb_tf, t_lb_tf, u_lb_tf,
                                  x_ub_tf, t_ub_tf, u_ub_tf)
    gradients = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss_value



lrs =[ 0.0001]
nlayers = [3]
#activ_funs= ["tf.nn.tanh", "tf.nn.relu"]
i=0
j=0
final_loss_val = []

for lr in range(len(lrs)):
    for nlayer in range(len(nlayers)):
        model = tf.keras.Sequential()
        j+=1
        layer = [2]+[32]*nlayers[nlayer] + [1]
        model = PINN(layer)
        model.summary()
        #if j>1:
        #    reinitialize(model)
        
        print("learning rate:", lrs[lr], "architecture:", layer)        # -------------------------------
        # Generate training data
        # -------------------------------

        # Number of points
        N_int = 1000  # Collocation (interior) points
        N_ic  = 200    # Initial condition points
        N_bc  = 200    # Boundary condition points

        # Domain bounds
        x_lower, x_upper = 0.0, np.pi
        t_lower, t_upper = 0.0, 1.0

        # Collocation points (interior of the domain)
        x_int = np.random.uniform(x_lower, x_upper, (N_int, 1))
        t_int = np.random.uniform(t_lower, t_upper, (N_int, 1))

        # Initial condition: u(x,0) = 0
        x_ic = np.random.uniform(x_lower, x_upper, (N_ic, 1))
        t_ic = np.zeros((N_ic, 1))
        u_ic = np.zeros((N_ic, 1))

        # Boundary conditions:
        # Left boundary: x = 0, u(0,t) = 1 - exp(-t)
        t_bc = np.random.uniform(t_lower, t_upper, (N_bc, 1))
        x_lb = np.zeros((N_bc, 1))
        u_lb = 1 - np.exp(-t_bc)

        # Right boundary: x = pi, u(pi,t) = -1 + exp(-t)
        x_ub = np.pi * np.ones((N_bc, 1))
        u_ub = -1 + np.exp(-t_bc)

        # Convert all data to TensorFlow tensors (float32)
        x_int_tf = tf.convert_to_tensor(x_int, dtype=tf.float32)
        t_int_tf = tf.convert_to_tensor(t_int, dtype=tf.float32)
        x_ic_tf  = tf.convert_to_tensor(x_ic, dtype=tf.float32)
        t_ic_tf  = tf.convert_to_tensor(t_ic, dtype=tf.float32)
        u_ic_tf  = tf.convert_to_tensor(u_ic, dtype=tf.float32)
        x_lb_tf  = tf.convert_to_tensor(x_lb, dtype=tf.float32)
        t_lb_tf  = tf.convert_to_tensor(t_bc, dtype=tf.float32)   # note: using t_bc for both boundaries
        u_lb_tf  = tf.convert_to_tensor(u_lb, dtype=tf.float32)
        x_ub_tf  = tf.convert_to_tensor(x_ub, dtype=tf.float32)
        t_ub_tf  = tf.convert_to_tensor(t_bc, dtype=tf.float32)
        u_ub_tf  = tf.convert_to_tensor(u_ub, dtype=tf.float32)

        # -------------------------------
        # Set up the optimizer and training loop
        # -------------------------------

        optimizer = tf.keras.optimizers.Adam(learning_rate=lrs[lr])
        Loss = []
        # Train the model
        epochs = 20001
        for epoch in range(epochs):
            loss_val = train_step()
            if epoch % 10 == 0:
                Loss.append(loss_val.numpy())
            if epoch % 1000 == 0:
                print(f"Epoch {epoch:05d}: Loss = {loss_val.numpy():.6e}")
            if epoch % 10000 == 0:
                final_loss_val.append(loss_val.numpy())  
        """          
        t_plot_loss = np.linspace(0, epochs,len(Loss))
        N_plot = 100
        x_plot = np.linspace(x_lower, x_upper, N_plot)
        t_plot = np.linspace(t_lower, t_upper, N_plot)
        X, T = np.meshgrid(x_plot, t_plot)
        X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
        #t_plot_loss = np.linspace(t_lower, t_upper,len(Loss))
        # Get the predicted u(x,t)
        u_pred = model(tf.convert_to_tensor(X_star, dtype=tf.float32))
        u_pred = u_pred.numpy().reshape(N_plot, N_plot)
        u_exact = np.cos(x_plot)- np.cos(x_plot)*np.exp(-1)
        plt.figure(num=2, figsize=(8,6))
        plt.subplot(len(lrs),len(nlayers),j)
        contour = plt.contourf(X, T, u_pred, levels=50, cmap='jet')
        plt.colorbar(contour)
        plt.xlabel("x")
        plt.ylabel("t")
        plt.title("Predicted solution u(x,t)")

        plt.figure(num=1, figsize=(6,4))
        plt.subplot(len(lrs),len(nlayers),j)
        plt.plot(t_plot_loss, Loss )
        plt.grid(True)
        plt.xlabel("epochs", fontdict= fontdic)
        plt.ylabel("Loss", fontdict= fontdic)
        plt.yscale("log")
        plt.title(f"Loss evolution with learning rate:{lrs[lr]} and number of layer: {nlayers[nlayer]}", fontdict= fontdic)
        plt.savefig(f"hyerpara_analysis_lr_{lrs[lr]}_nlyaer_{nlayers[nlayer]}_doubleepochs.png")
        #plt.show()
        del model
        K.clear_session()
        """                                 
#optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

#print(final_loss_val)
# -------------------------------
# Evaluate and plot the solution
# -------------------------------

N_plot = 100
x_plot = np.linspace(x_lower, x_upper, N_plot)
t_plot = np.linspace(t_lower, t_upper, N_plot)
X, T = np.meshgrid(x_plot, t_plot)
X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
t_plot_loss = np.linspace(t_lower, epochs,len(Loss))
# Get the predicted u(x,t)
u_pred = model(tf.convert_to_tensor(X_star, dtype=tf.float32))
u_pred = u_pred.numpy().reshape(N_plot, N_plot)
u_exact = np.cos(X)- np.cos(X)*np.exp(-T)
u_exact_end = np.cos(x_plot)- np.cos(x_plot)*np.exp(-1)
plt.figure(num = 1, figsize=(12,10))
plt.subplot(2,2,1)
contour = plt.contourf(X, T, u_pred, levels=50, cmap='jet')
plt.colorbar(contour)
plt.xlabel("x", fontdict= fontdic)
plt.ylabel("t", fontdict= fontdic)
plt.title("Predicted solution u(x,t)", fontdict= fontdic)
plt.subplot(2,2,3)
plt.plot(x_plot, u_pred[N_plot-1,:], label="u_pred" )
plt.plot(x_plot,u_exact_end, label="u_exact")
plt.xlabel("x", fontdict= fontdic)
plt.ylabel("u(x,1)", fontdict= fontdic)
plt.legend()
plt.title("Predicted solution u(x,1) and exact solution", fontdict= fontdic)
plt.subplot(2,2,4)
plt.plot(x_plot, np.abs(u_pred[N_plot-1, :]-u_exact_end))
plt.xlabel("x", fontdict= fontdic)
plt.ylabel("error", fontdict= fontdic)
plt.title("Error between predicted and exact solution at t=1", fontdict= fontdic)
"""
plt.subplot(2,2,5)
plt.plot(t_plot_loss, Loss )
plt.xlabel("epochs", fontdict= fontdic)
plt.ylabel("Loss", fontdict= fontdic)
plt.yscale("log")
plt.title("Loss evolution", fontdict= fontdic)
"""
plt.subplot(2,2,2)
contour = plt.contourf(X, T, np.abs(u_pred-u_exact), levels=50, cmap='viridis')
plt.colorbar(contour)
#plt.plot(t_plot_loss, Loss )
plt.xlabel("x", fontdict= fontdic)
plt.ylabel("t", fontdict= fontdic)
#plt.yscale("log")
plt.title("Error global", fontdict= fontdic)
plt.savefig("complete_validation_bc_weighting.png")
fig = plt.figure(num = 2, figsize=(10,8))
ax = plt.axes(projection ='3d')

surf = ax.plot_surface(X, T, np.abs(u_pred-u_exact), cmap='viridis')
fig.colorbar(surf)
avg_err_val=np.sum(np.abs(u_pred-u_exact))/(N_plot*N_plot)
plt.xlabel("x", fontdict= fontdic)
plt.ylabel("t", fontdict= fontdic)
plt.title(f"Error surface with average error value {avg_err_val:04e}")
print("average error value:", avg_err_val)
plt.savefig("error_surface.png")
fig = plt.figure(num = 3, figsize=(10,8))
ax = plt.axes(projection ='3d')

surf = ax.plot_surface(X, T, u_pred, cmap='viridis')
fig.colorbar(surf)
#avg_err_val=np.sum(np.abs(u_pred-u_exact))/(N_plot*N_plot)
plt.xlabel("x", fontdict= fontdic)
plt.ylabel("t", fontdict= fontdic)
#plt.title("Solution surface ")
#print("average error value:", avg_err_val)
plt.savefig("solution_surface.png")
plt.show()