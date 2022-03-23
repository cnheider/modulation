import numpy
from matplotlib import pyplot
from matplotlib.animation import FuncAnimation

pyplot.rcParams["figure.figsize"] = (8, 8)
pyplot.rcParams["font.size"] = 14
pyplot.style.use("dark_background")  # dark theme


def calculate_phase_from_focus(x, y, e):
    return numpy.sqrt(numpy.sum((e.r - numpy.array([x, y])) ** 2)) * (
        2 * numpy.pi / e.lambda0
    )


class Emitter:
    def __init__(self, x, y, c, f, phi, rMax=100, color="tab:blue", alpha=0.6):
        self.r, self.c, self.f, self.rMax, self.alpha = (
            numpy.array([x, y]),
            c,
            f,
            rMax,
            alpha,
        )
        self.color = color
        self.set_up()
        self.set_phase(phi)

    def increment(self, dt):
        self.t += dt
        if self.t < self.t0:
            return
        for i, circle in enumerate(self.circles):
            r = i * self.lambda0 + self.wrap(
                self.lambda0 * self.phi / (2 * numpy.pi) + self.c * self.t, self.lambda0
            )
            circle.set_height(2 * r)
            circle.set_width(2 * r)
            circle.set_alpha(self.alpha if i < ((self.t - self.t0) / self.T) else 0)

    def set_phase(self, phi):
        self.phi = self.wrap(phi, 2 * numpy.pi)
        self.t0 = self.T * (1 - self.phi / (2 * numpy.pi))
        self.t = 0

    def set_up(self):
        self.lambda0 = self.c / self.f
        self.T = 1.0 / self.f
        self.N = numpy.int32(numpy.ceil(self.rMax / self.lambda0))
        self.circles = [
            pyplot.Circle(
                xy=tuple(self.r),
                fill=False,
                lw=2,
                radius=0,
                alpha=self.alpha,
                color=self.color,
            )
            for i in range(self.N)
        ]

    def wrap(self, x, x_max):
        if x >= 0:
            return x - numpy.floor(x / x_max) * x_max
        if x < 0:
            return x_max - (-x - numpy.floor(-x / x_max) * x_max)


class EmitterArray:
    def __init__(self):
        self.emitters = []

    def add_emitter(self, e):
        self.emitters.append(e)

    def increment(self, dt):
        for emitter in self.emitters:
            emitter.increment(dt)

    def get_circles(self):
        """Get all the circles from all the emitters"""
        circles = []
        for emitter in self.emitters:
            circles.extend(emitter.circles)
        return circles

    def remove_offset(self):
        """Only run this one time after all emitters have been added"""
        offsets = []
        for emitter in self.emitters:
            offsets.append(emitter.t0)
        offset_min = numpy.min(offsets)
        for emitter in self.emitters:
            emitter.increment(offset_min)

    @property
    def circles(self):
        return self.get_circles()


def asd():
    FPS = 30
    X, Y = 100, 100
    c, f = 3, 0.2
    lambda0 = c / f

    N = 10

    emitter_array = EmitterArray()

    # ########################################################
    # # DEMO 1 - Linear Array of Emitters
    # xs = numpy.linspace(-lambda0/4, lambda0/4, N)
    # ys = numpy.zeros_like(xs)
    # phi = numpy.linspace(0,numpy.pi/2,N)
    # for i in range(N):
    #     e = Emitter(xs[i], ys[i], c, f, phi[i])
    #     emitter_array.AddEmitter(e)
    # #######################################################

    # ########################################################
    # # DEMO 2 - Linear Array of Emitters
    # r = numpy.linspace(-lambda0/4, lambda0/4, N)
    # angle = numpy.pi/4
    # xs = r*numpy.cos(angle)
    # ys = r*numpy.sin(angle)
    # phi = numpy.linspace(0 , numpy.pi/2, N)
    # for i in range(N):
    #     e = Emitter(xs[i], ys[i], c, f, phi[i])
    #     emitter_array.AddEmitter(e)
    # #######################################################

    # ########################################################
    # # DEMO 3 - Focussed Array
    xs = numpy.linspace(-lambda0, lambda0, N)
    ys = numpy.zeros_like(xs)
    for i in range(N):
        e = Emitter(xs[i], ys[i], c, f, 0)
        phase = calculate_phase_from_focus(0, 20, e)
        e.set_phase(phase)
        emitter_array.add_emitter(e)
    # #######################################################

    # ########################################################
    # # DEMO 4 - Dual Frequency Emitters
    # xs = numpy.linspace(-lambda0/4, lambda0/4, N)
    # ys = numpy.zeros_like(xs)
    # phi = numpy.linspace(0,numpy.pi/2,N)
    # for i in range(N):
    #     e = Emitter(xs[i], ys[i], c, f, phi[i])
    #     emitter_array.AddEmitter(e)

    # for i in range(N):
    #     e = Emitter(xs[i], ys[i], c, 0.5*f, -phi[i], color = "red")
    #     emitter_array.AddEmitter(e)
    # #######################################################

    # # ########################################################
    # # # DEMO 5 - Focussed Array
    # xs = numpy.linspace(-lambda0, lambda0, N)
    # ys = numpy.zeros_like(xs)
    # for i in range(N):
    #     e = Emitter(xs[i], ys[i], c, f, 0)
    #     phase = CalculatePhaseFromFocus(0, 20, e)
    #     e.SetPhase(phase)
    #     emitter_array.AddEmitter(e)

    # for i in range(N):
    #     e = Emitter(xs[i], ys[i], c, 0.8*f, 0, color = "red")
    #     phase = CalculatePhaseFromFocus(-20, 30, e)
    #     e.SetPhase(phase)
    #     emitter_array.AddEmitter(e)
    # # #######################################################

    # # # ########################################################
    # # # # DEMO 6 - Focussed Array Random
    # xs = numpy.random.uniform(-lambda0/2, lambda0/2, N)
    # ys = numpy.random.uniform(-lambda0/2, lambda0/2, N)
    # for i in range(N):
    #     e = Emitter(xs[i], ys[i], c, f, 0)
    #     phase = CalculatePhaseFromFocus(0, 20, e)
    #     e.SetPhase(phase)
    #     emitter_array.AddEmitter(e)
    # # # #######################################################

    # # # ########################################################
    # # # # DEMO 7 - Focussed Array Random
    # N = 20
    # xs = numpy.random.uniform(-lambda0*2, lambda0*2, N)
    # ys = numpy.random.uniform(-lambda0*2, lambda0*2, N)
    # for i in range(N):
    #     e = Emitter(xs[i], ys[i], c, f, 0)
    #     phase = CalculatePhaseFromFocus(0, 20, e)
    #     e.SetPhase(phase)
    #     emitter_array.AddEmitter(e)
    # # # #######################################################

    emitter_array.remove_offset()

    fig, ax = pyplot.subplots()
    ax.set_xlim([-X / 2, Y / 2])
    ax.set_ylim([-X / 2, Y / 2])
    ax.set_aspect(1)
    ax.grid(alpha=0.2)
    fig.tight_layout()

    for circle in emitter_array.circles:
        ax.add_patch(circle)

    for emitter in emitter_array.emitters:
        ax.add_patch(pyplot.Circle(tuple(emitter.r), 0.4, color="purple"))

    def init():
        return tuple(emitter_array.circles)

    def update(frame_number):
        emitter_array.increment(1 / FPS)
        return tuple(emitter_array.circles)

    ani = FuncAnimation(fig, update, init_func=init, interval=1000 / FPS, blit=True)
    pyplot.show()


if __name__ == "__main__":
    asd()
