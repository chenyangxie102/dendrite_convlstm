
import math
import shutil

import numpy as np
import matplotlib.pyplot as plt
import os

# Simulation cell parameters:
Nx = 300            # Number of grid points in the x-direction.
Ny = 300            # Number of grid points in the y-direction.
NxNy = Nx*Ny        # Total number of grid points in the simulation cell.
dx = 0.03           # Grid spacing between two grid points in the x-direction.
dy = 0.03           # Grid spacing between two grid points in the y-direction.

# Time integration parameters:
nstep = 4001        # Number of time integration steps.
nprint = 80        # Output frequency to write the results to file.
dtime = 1.0e-4      # Time increment for numerical integration.

# Material specific parameters:
tau = 0.0003
epsilonb = 0.01     #epsilon的平均值，epsilon依赖外界法向量方向
mu = 1.0
# kappa = 1.6         #a dimensionless latent heat,无量纲潜热
kappa_range = range(111,221,1)    #到时候还要除以100
delta = 0.02        #the strength of the anisotropy,各向异性强度
aniso = 4           #the mode number of anisotropy,各向异性的模态数
alpha = 0.9         #正常数
gamma = 10.0
teq = 1.0           #平衡温度
theta0 = 0.2        #初始偏移角
seed = 4.0          # The size of the initial seed.是种子大小，而不是种子数量


################################################################################
# Plot colormap
# def plot(data, step):
#
#
#     plt.figure(figsize=(10, 10))
#     plt.pcolormesh(data, cmap='gray')
#     plt.axis('off')
#     path_image = 'images/sample_15/'
#     # if os.path.exists(path_image):
#     #     shutil.rmtree(path_image)
#     #     os.makedirs(path_image)
#     # else:
#     #     os.makedirs(path_image)
#
#     plt.savefig(path_image +'img_' + str(step) + '.png', bbox_inches='tight', pad_inches=0)
#     plt.close()
#     # plt.savefig( 'img_' + str(step) + '.png', bbox_inches='tight', pad_inches=0)
#     # if os.path.exists(path):
#     #     shutil.rmtree(path)
#     #     os.makedirs(path)
#     # else:
#     #     os.makedirs(path)


# Plot colormap
def plot(data, step,path_image):
    if not os.path.exists(path_image):
        os.makedirs(path_image)

    plt.figure(figsize=(10, 10))
    plt.pcolormesh(data, cmap='gray')
    plt.axis('off')
    #sample_value
    plt.savefig(path_image +'img_' + str(step) + '.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    # plt.savefig( 'img_' + str(step) + '.png', bbox_inches='tight', pad_inches=0)
    # if os.path.exists(path):
    #     shutil.rmtree(path)
    #     os.makedirs(path)
    # else:
    #     os.makedirs(path)



################################################################################
# This function calculates the two-dimensional laplacians for a
# given array with five-point finite difference algorithm.
def lap(array, Nx, Ny, dx, dy):
    laplace = np.zeros((Nx, Ny), dtype=float)
    for i in range(Nx):
        for j in range(Ny):
            # Periodic boundary conditions:
            jp = j+1
            jm = j-1
            ip = i+1
            im = i-1

            if i == 0:
                im = Nx - 1
            if i == (Nx-1):
                ip = 0

            if j == 0:
                jm = Ny-1
            if j == (Ny-1):
                jp = 0
            # Laplace operator with five point stencil
            laplace[i, j] = (array[ip, j] - 2*array[i, j] + array[im, j])/dx**2 \
                + (array[i, jp] - 2*array[i, j] + array[i, jm])/dy**2
    return(laplace)


################################################################################
# These two functions calculate the gradient components in the
# x- and y-directions.
def gradx(array, Nx, Ny, dx, dy):
    gradientx = np.zeros((Nx, Ny), dtype=float)
    for i in range(Nx):
        for j in range(Ny):
            # Periodic boundary conditions:
            jp = j+1
            jm = j-1
            ip = i+1
            im = i-1

            if i == 0:
                im = Nx - 1
            if i == (Nx-1):
                ip = 0

            if j == 0:
                jm = Ny-1
            if j == (Ny-1):
                jp = 0
            # Derivatives with the centered difference:
            gradientx[i, j] = (array[ip, j] - array[im, j])/(2.0*dx)
    return(gradientx)


def grady(array, Nx, Ny, dx, dy):
    gradienty = np.zeros((Nx, Ny), dtype=float)
    for i in range(Nx):
        for j in range(Ny):
            # Periodic boundary conditions:
            jp = j+1
            jm = j-1
            ip = i+1
            im = i-1

            if i == 0:
                im = Nx - 1
            if i == (Nx-1):
                ip = 0

            if j == 0:
                jm = Ny-1
            if j == (Ny-1):
                jp = 0
            # Derivatives with the centered difference:
            gradienty[i, j] = (array[i, jp] - array[i, jm])/(2.0*dy)
    return(gradienty)


################################################################################
# # Introduce the initial nuclei:
# phi = np.zeros((Nx, Ny), dtype=float)
# tem = np.zeros((Nx, Ny), dtype=float)
#
# for i in range(Nx):
#     for j in range(Ny):
#         if (i-Nx/2)**2 + (j-Ny/2)**2 < seed:
#             phi[i, j] = 1.0


################################################################################
# Main program: time integration

# for istep in range(nstep):
#     # Assign current phi to phiold
#     phiold = phi
#
#     # Calculate the laplacians of order parameter phi and the temperature tem
#     # for all grid points
#     lap_phi = lap(phi, Nx, Ny, dx, dy)
#     lap_tem = lap(tem, Nx, Ny, dx, dy)
#
#     # Calculate the gradient of Phi for all grid points
#     phidx = gradx(phi, Nx, Ny, dx, dy)
#     phidy = grady(phi, Nx, Ny, dx, dy)
#
#     # Calculate angles for all grid points
#     theta = np.arctan2(phidx, phidy)
#
#     # Calculate epsilon and its derivative
#     epsilon = epsilonb * (1.0 + delta * np.cos(aniso * (theta - theta0)))
#     epsilon_deriv = -epsilonb * delta * aniso * np.sin(aniso * (theta - theta0))
#
#     # Calculate the first term of TDGL equation
#     dummyx = epsilon * epsilon_deriv * phidx
#     term1 = grady(dummyx, Nx, Ny, dx, dy)
#
#     # Calculate the second term of TDGL equation
#     dummyy = -epsilon * epsilon_deriv * phidy
#     term2 = gradx(dummyy, Nx, Ny, dx, dy)
#
#     # Calculate the value of m
#     m = (alpha/math.pi) * np.arctan(gamma * (teq - tem))
#
#     # Evolve phi with Euler time integration
#     phi = phi + (dtime/tau) * (term1 + term2 + np.square(epsilon) * lap_phi \
#         + phi * (1.0 - phi) * (phi - 0.5 + m))
#
#     # Evolve temperature with Euler time integration
#     tem = tem + dtime * lap_tem + kappa * (phi - phiold)
#
#     # Print results
#     if istep % nprint == 0:
#         print(istep)
#
#         # np.savetxt('phi' + str(istep) + '.txt', phi)
#         plot(phi, istep)





########################################

# Main program: time integration
for kappa_value in kappa_range:
    kappa = kappa_value/100
    sample_value = int(kappa_value - 110)
    print('kappa值：',kappa)
    print('绘制图片sample_',sample_value)

    # Introduce the initial nuclei:
    phi = np.zeros((Nx, Ny), dtype=float)
    tem = np.zeros((Nx, Ny), dtype=float)

    for i in range(Nx):
        for j in range(Ny):
            if (i - Nx / 2) ** 2 + (j - Ny / 2) ** 2 < seed:
                phi[i, j] = 1.0

    for istep in range(nstep):
        # Assign current phi to phiold
        phiold = phi

        # Calculate the laplacians of order parameter phi and the temperature tem
        # for all grid points
        lap_phi = lap(phi, Nx, Ny, dx, dy)
        lap_tem = lap(tem, Nx, Ny, dx, dy)

        # Calculate the gradient of Phi for all grid points
        phidx = gradx(phi, Nx, Ny, dx, dy)
        phidy = grady(phi, Nx, Ny, dx, dy)

        # Calculate angles for all grid points
        theta = np.arctan2(phidx, phidy)

        # Calculate epsilon and its derivative
        epsilon = epsilonb * (1.0 + delta * np.cos(aniso * (theta - theta0)))
        epsilon_deriv = -epsilonb * delta * aniso * np.sin(aniso * (theta - theta0))

        # Calculate the first term of TDGL equation
        dummyx = epsilon * epsilon_deriv * phidx
        term1 = grady(dummyx, Nx, Ny, dx, dy)

        # Calculate the second term of TDGL equation
        dummyy = -epsilon * epsilon_deriv * phidy
        term2 = gradx(dummyy, Nx, Ny, dx, dy)

        # Calculate the value of m
        m = (alpha/math.pi) * np.arctan(gamma * (teq - tem))

        # Evolve phi with Euler time integration
        phi = phi + (dtime/tau) * (term1 + term2 + np.square(epsilon) * lap_phi \
            + phi * (1.0 - phi) * (phi - 0.5 + m))
        # Evolve temperature with Euler time integration
        tem = tem + dtime * lap_tem + kappa * (phi - phiold)


        # Print results

        if istep % nprint == 0:
            print(istep)
            # np.savetxt('phi' + str(istep) + '.txt', phi)
            path_image = f'images_1/sample_{sample_value}/'

            plot(phi, istep,path_image)


    #     #     # Print results
    # #     print('计算准备开始：')
    # if istep % nprint == 0:
    # #     print(istep)
    # #     # np.savetxt('phi' + str(istep) + '.txt', phi)
    # #     sample_value = int(kappa * 100 - 109)
    # #     print("开始绘图：")
    # plot(phi, istep, sample_value)
    #