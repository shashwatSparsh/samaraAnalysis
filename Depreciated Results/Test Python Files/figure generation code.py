fig1, ax1 = plt.subplots(dpi=800, figsize=(12,8))
ax1.plot(tNorm, xNorm, label='Normalized X Position')
ax1.plot(tNorm, yNorm, label='Normalized Y Position')
ax1.set(title="Normalized Positions over Time",
        xlabel="Time[s]",
        ylabel='Position')
#plt.legend()
box = ax1.get_position()
ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
# Put a legend below current axis
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=True, shadow=False, ncol=5)
ax1.grid(True)

fig2, (ax1, ax2, ax3) = plt.subplots(3, dpi=800, figsize=(12,8))
ax1.plot(vTime, vY, label='Descent Speed')
ax1.set(title="Descent Speed over time",
        xlabel="Time [s]",
        ylabel="Descent Speed [px/s]")
ax2.plot(vTime, vYsmooth, label='Smooth Descent Speed')
ax2.set(title="Descent Speed with Kalman Filter",
        xlabel="Time [s]",
        ylabel="Descent Speed [px/s]")
ax3.plot(vTime, vYsmoothMPS, label='Smooth Descent Speed')
ax3.set(title="Descent Speed over time",
        xlabel="Time [s]",
        ylabel="Descent Speed [meters/s]")
ax1.grid(True)
ax2.grid(True)
ax3.grid(True)
fig2.tight_layout()

fig3, ax1 = plt.subplots(dpi=800, figsize=(12,8))
ax1.plot(aTime, aYMPS2, label='Net Acceleration')
ax1.set_title("Net Acceleration [m/$\mathregular{s^2}$]")
ax1.set(xlabel="Time [s]", ylabel='Speed [m/s]')
ax1.grid(True)

fig4, ax1 = plt.subplots(dpi=800, figsize=(12,8))
ax1.plot(vTime, vYsmoothMPS, label='Descent Speed over time')
ax1.set(title="Descent Velocity over Time",
        xlabel='Time [s]',
        ylabel='Descent Speed [meters/s]')
ax1.grid(True)