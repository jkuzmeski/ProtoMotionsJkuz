# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np

import mpl_toolkits.mplot3d  # noqa: F401


class DebugMotionViewer:
    """
    Helper class to visualize motion data from NumPy-file format.
    """

    def __init__(self, motion_file: str, render_scene: bool = False) -> None:
        """Load a motion file and initialize the internal variables.
        Args:
            motion_file: Motion file path to load.
            render_scene: Whether the scene (space occupied by the skeleton during movement)
                is rendered instead of a reduced view of the skeleton.
        Raises:
            AssertionError: If the specified motion file doesn't exist.
        """
        self._figure = None
        self._figure_axes = None
        self._render_scene = render_scene

        # load motions
        data = np.load(motion_file, allow_pickle=True)
        self._body_positions = data["body_positions"]
        self._body_names = data["body_names"]
        self._dt = 1.0 / data["fps"]

        self._num_frames = self._body_positions.shape[0]
        self._current_frame = 0

        print("\nBody")
        for i, name in enumerate(self._body_names):
            minimum = np.min(self._body_positions[:, i], axis=0).round(decimals=2)
            maximum = np.max(self._body_positions[:, i], axis=0).round(decimals=2)
            print(f"  |-- [{name}] minimum position: {minimum}, maximum position: {maximum}")

    def _drawing_callback(self, frame: int) -> None:
        """Drawing callback called each frame"""
        # get current motion frame
        # get data
        vertices = self._body_positions[self._current_frame]
        # draw skeleton state
        self._figure_axes.clear()
        self._figure_axes.scatter(*vertices.T, color="black", depthshade=False)

        # add labels for each body part
        for i, (position, name) in enumerate(zip(vertices, self._body_names)):
            # offset the text slightly to avoid overlapping with the point
            self._figure_axes.text(
                position[0] + 0.05, position[1] + 0.05, position[2] + 0.05, 
                name, 
                fontsize=8, 
                alpha=0.8
            )
        # adjust exes according to motion view
        # - scene
        if self._render_scene:
            # compute axes limits
            minimum = np.min(self._body_positions.reshape(-1, 3), axis=0)
            maximum = np.max(self._body_positions.reshape(-1, 3), axis=0)
            center = 0.5 * (maximum + minimum)
            diff = 0.75 * (maximum - minimum)
        # - skeleton
        else:
            # compute axes limits
            minimum = np.min(vertices, axis=0)
            maximum = np.max(vertices, axis=0)
            center = 0.5 * (maximum + minimum)
            diff = np.array([0.75 * np.max(maximum - minimum).item()] * 3)
        # scale view
        self._figure_axes.set_xlim((center[0] - diff[0], center[0] + diff[0]))
        self._figure_axes.set_ylim((center[1] - diff[1], center[1] + diff[1]))
        self._figure_axes.set_zlim((center[2] - diff[2], center[2] + diff[2]))
        self._figure_axes.set_box_aspect(aspect=diff / diff[0])
        # plot ground plane
        x, y = np.meshgrid([center[0] - diff[0], center[0] + diff[0]], [center[1] - diff[1], center[1] + diff[1]])
        self._figure_axes.plot_surface(x, y, np.zeros_like(x), color="green", alpha=0.2)
        # print metadata
        self._figure_axes.set_xlabel("X")
        self._figure_axes.set_ylabel("Y")
        self._figure_axes.set_zlabel("Z")
        self._figure_axes.set_title(f"frame: {self._current_frame}/{self._num_frames}")
        # increase frame counter
        self._current_frame += 1
        if self._current_frame >= self._num_frames:
            self._current_frame = 0

    def show(self) -> None:
        """Show motion"""
        # create a 3D figure
        self._figure = plt.figure()
        self._figure_axes = self._figure.add_subplot(projection="3d")
        # matplotlib animation (the instance must live as long as the animation will run)
        self._animation = matplotlib.animation.FuncAnimation(
            fig=self._figure,
            func=self._drawing_callback,
            frames=self._num_frames,
            interval=1000 * self._dt,
        )
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Motion file")
    parser.add_argument(
        "--render-scene",
        action="store_true",
        default=False,
        help=(
            "Whether the scene (space occupied by the skeleton during movement) is rendered instead of a reduced view"
            " of the skeleton."
        ),
    )
    parser.add_argument("--matplotlib-backend", type=str, default="TkAgg", help="Matplotlib interactive backend")
    args, _ = parser.parse_known_args()

    # https://matplotlib.org/stable/users/explain/figure/backends.html#interactive-backends
    matplotlib.use(args.matplotlib_backend)

    viewer = DebugMotionViewer(args.file, render_scene=args.render_scene)
    viewer.show()
