import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from PIL.Image import Image


def plot_heat_over_time(image_path, num_iterations=100, dt=0.2):
    def one_step_heat(u, dt, dx=1, dy=1, alpha=None):
        u_n = u.copy()
        u = np.pad(u, 1, 'constant', constant_values=0)
        u = u.astype(np.float64)
        u_n = u_n.astype(np.float64)

        # explicitly the code can be implemented as follows:
        # for i in range(u_n.shape[0]):
        #     i_u = i + 1
        #     for j in range(u_n.shape[1]):
        #         j_u = j + 1
        #         u_n[i, j] = u[i_u, j_u] + dt * (
        #                 ((u[i_u, j_u] -2*u[i_u, j_u] + u[i_u-1, j_u])
        #                  /dx**2)
        #                 +
        #                 ((u[i_u, j_u+1] -2*u[i_u, j_u] + u[i_u, j_u-1])
        #                  /dy**2)
        #         )

        # but it is too slow, so we use numpy broadcasting to speed it up  :)
        u_n += dt * (
                (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1])
                / (dx ** 2)
                +
                (+ u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2])
                / (dy ** 2)
        )
        return u_n

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_arrray = []
    image_arrray.append(image.astype(np.uint8))
    for i in tqdm(range(1, num_iterations)):
        image_arrray.append(one_step_heat(image_arrray[i - 1], dt).astype(np.uint8))
        if np.array_equal(image_arrray[i], image_arrray[i - 1]):
            print(f'converged after {i} iterations')
            break
    # make a video from the image array
    # save images to file
    size = image.shape
    fps = 25
    out = cv2.VideoWriter(f'heatequation_video_{dt}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]),
                          False)
    for i in range(len(image_arrray)):
        # data = np.random.randint(0, 256, size, dtype='uint8')
        data = image_arrray[i]
        out.write(data)
    out.release()


def linear_2d_heat_equation(u, dt=1000000, dx=1, dy=1, alpha=None):
    """
    # apply the 2d linear heat equation u_t = u_xx + u_yy to the image u(x,y;0) = I(x,y)
    # Program either the explicit form with forward Euler numerical iterations in time and central derivative approximation in space,
    # or by a direct multiplication in the frequency domain by multiplying the transformed signal with the transformed gaussian kernel.
    # What is the transform of the gaussian kernel?
    # for the explicit form, denote U(i,j;n) = u(i_dx, j_dy; n_dt);
    # then the iterative update scheme is giveb by:
    # D+(t)U(i,j;n) = (D+(x)D_(x) + D+(y)D_(y))U(i,j;n)
    # or explicitly:
    # U(i,j;n+1) = U(i,j;n) + dt * ( ((U(i+1,j;n) -2U(i,j;n) + U(i-1,j;n))/dx^2) + ((U(i,j+1;n) -2U(i,j;n) + U(i,j-1;n))/dy^2) )

    :param u: image to apply the heat equation to
    :param dt: time step
    :param dx: space step in x
    :param dy: space step in y
    :param alpha: alpha parameter, currently not used
    :return: the image after applying the heat equation
    """

    print(u.shape)
    u = np.pad(u, 1, 'constant', constant_values=0)
    print(u.shape)

    # replace the padding with something interesting
    u[0, ::2] = 255  # top row
    u[:-1, ::2] = 255  # bottom row
    u[::2, 0] = 255  # left column
    u[::2, -1] = 255  # right column
    print(u[0, :])
    u_n = u[1:, :-1].copy()
    print(u_n.shape)
    unchanged = False
    while not unchanged:
        for i in range(u_n.shape[0]):
            for j in range(u_n.shape[1]):
                u_n[i, j] = u[i, j] + dt * (
                        ((u[i + 1, j] - 2 * u[i, j] + u[i - 1, j])
                         / dx ** 2)
                        +
                        ((u[i, j + 1] - 2 * u[i, j] + u[i, j - 1])
                         / dy ** 2)
                )
        if np.array_equal(u, u_n):
            unchanged = True
        # u = u_n.copy()
        u[1:, :-1] = u_n.copy()
        print(f'avrg: {np.average(u)}')
        plt.imshow(u_n, cmap='gray', interpolation=None)
        plt.show()

    return u


class HeatEquationDataser():
    """
    this dataset is initialized with a list image sequences. each sequence starts with a
    random image, and then the heat equation is applied recursively.
    """

    def __init__(self, num_frames, starting_images=None, image_size=(64, 64), num_sequences=10, dt=1000000, dx=1, dy=1,
                 alpha=None):
        print("init dataset")
        if starting_images is None:
            starting_images = []
            for i in range(num_sequences):
                starting_images.append(np.random.randint(0, 255, image_size))
        else:
            self.image_size = starting_images[0].shape
        self.dt = dt
        self.data = []
        for i in range(len(starting_images)):
            sequence = np.zeros((num_frames, *self.image_size))
            sequence[0] = starting_images[i]
            for j in range(1, num_frames):
                sequence[j] = self._one_step_heat(sequence[j - 1], dt=self.dt)
            self.data.append(sequence)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _one_step_heat(u, dt=0.2, dx=1, dy=1, alpha=None):
        u_n = u.copy()
        u = np.pad(u, 1, 'constant', constant_values=0)

        for i in range(u_n.shape[0]):
            i_u = i + 1
            for j in range(u_n.shape[1]):
                j_u = j + 1
                u_n[i, j] = u[i_u, j_u] + dt * (
                        ((u[i_u, j_u] - 2 * u[i_u, j_u] + u[i_u - 1, j_u])
                         / dx ** 2)
                        +
                        ((u[i_u, j_u + 1] - 2 * u[i_u, j_u] + u[i_u, j_u - 1])
                         / dy ** 2)
                )
        return u_n


class HeatEquationBinaryDataset(HeatEquationDataser):
    """
    this dataset is wrapping the HeatEquationDataset, and returns images in pairs.
    it creates the pairs from sequences in the HeatEquationDataset, by taking the following images.
    """

    def __init__(self, num_frames, starting_images=None, image_size=(64, 64), num_sequences=10):
        super().__init__(num_frames, starting_images, image_size, num_sequences)

    def __getitem__(self, index):
        sequence_index = index // (self.data[0].shape[0] - 1)
        frame_index = index % (self.data[0].shape[0] - 1)
        return self.data[sequence_index][frame_index], self.data[sequence_index][frame_index + 1]

    def __len__(self):
        return len(self.data * (self.data[0].shape[0] - 1))


def main():
    image_path = 'cameraman.png'
    plot_heat_over_time(image_path, num_iterations=10000, dt=0.001)
    exit(0)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    starting_images = [image,
                       np.random.randint(0, 255, image.shape)]

    dataset = HeatEquationBinaryDataset(num_frames=5, starting_images=starting_images, image_size=(64, 64),
                                        num_sequences=1)
    print(len(dataset))
    for i in range(len(dataset)):
        print(dataset[i][0].shape)
        print(dataset[i][1].shape)
        plt.imshow(dataset[i][0], cmap='gray', interpolation=None)
        plt.show()
        plt.imshow(dataset[i][1], cmap='gray', interpolation=None)
        plt.show()


if __name__ == '__main__':
    main()
