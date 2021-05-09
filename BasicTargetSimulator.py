'''
Geometric generation of lines & circles.
'''

import numpy as np

def hor_circle(times, acc, x0, v0, right):
    t = times - times[0]
    vx0 = v0[0]
    vy0 = v0[1]
    vz0 = v0[2]
    vxy = np.linalg.norm(v0[:2])
    R = vxy**2 / acc
    omega = vxy / R
    theta = (omega*t) % (2*np.pi)
    sign = (2*right-1) * (2*(vy0>=0)-1)

    if np.abs(vy0) > 1e-3:
            v_ratio = vx0 / vy0
            theta0 = np.arctan(v_ratio)
            fac = np.sqrt(1+(v_ratio)**2)
            x_cents = x0[0] + sign*R/fac
            y_cents = x0[1] - sign*R*v_ratio/fac
    else:
            theta0 = np.arctan(vx0/np.abs(vy0))
            x_cents = x0[0]
            y_cents = x0[1]-sign*R if vx0>=0 else x0[1]+sign*R

    theta = (theta-np.pi if right else -theta) if vy0>=0 else (theta if right else np.pi-theta)

    theta = theta0 + theta
    x = x_cents + R*np.cos(theta)
    dy = R*np.sin(theta)
    y = y_cents - dy
    z = x0[2] + t*vz0

    ids = np.abs(dy) <= 1e-5
    dy[ids] = 0
    vx_pos_sign = (dy<=0) if right else (dy>=0)
    vx_sign = 2 * vx_pos_sign - 1
    v_ratio = (x-x_cents) / (y-y_cents)

    vx = vx_sign * vxy / np.sqrt(1+v_ratio**2)
    vy = -vx_sign * vxy * v_ratio/np.sqrt((1+v_ratio**2))
    vz = v0[2] * np.ones(np.size(vy))

    return np.block([[x],[y],[z]]), np.block([[vx],[vy],[vz]])

def vert_circle(times,acc,x0,v0,up=True,phi_max=45,eps=1e-6):
    Vabs = np.linalg.norm(v0)
    sign = (-1)**(up+1)
    phi_max = phi_max/180*np.pi

    xx, xy, xz = [x0[0]], [x0[1]], [x0[2]]
    vx, vy, vz = [v0[0]], [v0[1]], [v0[2]]

    for i in range(1,len(times)):
        dt = times[i] - times[i-1]
        phi = np.arccos(vz[-1]/(Vabs+eps))
        phi_shifted = np.pi/2 - phi
        theta = np.arctan2(vy[-1], vx[-1])
        if (up and (phi_shifted >= phi_max)) or ((not up) and (phi_shifted <= -phi_max)):
            vx.append(vx[-1])
            vy.append(vy[-1])
            vz.append(vz[-1])
        else:
            vx_tmp = vx[-1] - sign * dt * acc*np.cos(phi)*np.cos(theta)
            vy_tmp = vy[-1] - sign * dt * acc*np.cos(phi)*np.sin(theta)
            vz_tmp = vz[-1] + sign * dt * acc*np.sin(phi)
            fac = np.sqrt(vx_tmp**2+vy_tmp**2+vz_tmp**2) / Vabs # don't let absolute velocity change
            vx.append(vx_tmp/fac)
            vy.append(vy_tmp/fac)
            vz.append(vz_tmp/fac)
        xx.append(xx[-1] + dt * 0.5*(vx[-2]+vx[-1]))
        xy.append(xy[-1] + dt * 0.5*(vy[-2]+vy[-1]))
        xz.append(xz[-1] + dt * 0.5*(vz[-2]+vz[-1]))

    return np.block([[np.array(xx)], [np.array(xy)], [np.array(xz)]]),\
           np.block([[np.array(vx)], [np.array(vy)], [np.array(vz)]])

def line(times, x0, v0, acc, vmax=None, theta=np.pi/2, phi=np.pi*2/3):
        t = times - times[0]

        vx = v0[0] + t*acc*np.sin(phi)*np.cos(theta)
        vy = v0[1] + t*acc*np.sin(phi)*np.sin(theta)
        vz = v0[2] + t*acc*np.cos(phi)

        V = np.block([[vx],[vy],[vz]])
        if vmax:
            vabs = np.sqrt(np.sum(V**2, axis=0))
            if vabs[0] < vmax:
                tmax = np.where(vabs>=vmax)[0]
                if len(tmax)>0:
                    V[:, tmax] = V[:, tmax] / vabs[tmax] * vmax
            else:
                # already faster than Vmax - just keep current speed
                vx, vy, vz = v0[0]+0.*t, v0[1]+0.*t, v0[2]+0.*t
                V = np.block([[vx],[vy],[vz]])

        dx = np.diff(t) * (V[:,1:] + V[:,:-1]) / 2
        dx_agg = np.cumsum(dx, axis=1)
        dx_agg = np.concatenate((np.zeros((3,1)), dx_agg), axis=1)
        X = np.array(x0).reshape((3,1)) + dx_agg

        return X, V
