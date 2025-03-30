import logging
import math

import cv2
import mujoco
import numpy as np
import numpy.typing as npt
from pyrr import Vector3

from revolve2.simulation.scene import Scene, SimulationState
from revolve2.simulation.simulator import RecordSettings

from ._control_interface_impl import ControlInterfaceImpl
from ._open_gl_vision import OpenGLVision
from ._render_backend import RenderBackend
from ._scene_to_model import scene_to_model
from ._simulation_state_impl import SimulationStateImpl
from .viewers import CustomMujocoViewer, NativeMujocoViewer, ViewerType


def simulate_scene(
    scene_id: int,
    scene: Scene,
    headless: bool,
    record_settings: RecordSettings | None,
    start_paused: bool,
    control_step: float,
    sample_step: float | None,
    simulation_time: int | None,
    simulation_timestep: float,
    cast_shadows: bool,
    fast_sim: bool,
    viewer_type: ViewerType,
    teleport_handlers: list = None,
    render_backend: RenderBackend = RenderBackend.EGL,
) -> list[SimulationState]:
    """
    Simulate a scene.

    :param scene_id: An id for this scene, unique between all scenes ran in parallel.
    :param scene: The scene to simulate.
    :param headless: If False, a viewer will be opened that allows a user to manually view and manually interact with the simulation.
    :param record_settings: If not None, recording will be done according to these settings.
    :param start_paused: If true, the simulation will start in a paused state. Only makessense when headless is False.
    :param control_step: The time between each call to the handle function of the scene handler. In seconds.
    :param sample_step: The time between each state sample of the simulation. In seconds.
    :param simulation_time: How long to simulate for. In seconds.
    :param simulation_timestep: The duration to integrate over during each step of the simulation. In seconds.
    :param cast_shadows: If shadows are cast.
    :param fast_sim: If fancy rendering is disabled.
    :param viewer_type: The type of viewer used for the rendering in a window.
    :param teleport_handlers: List of handlers for teleportation logic.
    :param render_backend: The backend to be used for rendering (EGL by default and switches to GLFW if no cameras are on the robot).
    :returns: The results of simulation. The number of returned states depends on `sample_step`.
    :raises ValueError: If the viewer is not able to record.
    """
    logging.info(f"Simulating scene {scene_id}")

    # Initialize if teleport_handlers is None
    if teleport_handlers is None:
        teleport_handlers = []

    """Define mujoco data and model objects for simuating."""
    model, mapping = scene_to_model(
        scene, simulation_timestep, cast_shadows=cast_shadows, fast_sim=fast_sim
    )
    data = mujoco.MjData(model)

    """Define a control interface for the mujoco simulation (used to control robots)."""
    control_interface = ControlInterfaceImpl(
        data=data, abstraction_to_mujoco_mapping=mapping
    )
    """Make separate viewer for camera sensors."""
    camera_viewers = {
        camera.camera_id: OpenGLVision(
            model=model, camera=camera, headless=headless, open_gl_lib=render_backend
        )
        for camera in mapping.camera_sensor.values()
    }

    """Define some additional control variables."""
    last_control_time = 0.0
    last_sample_time = 0.0
    last_video_time = 0.0  # time at which last video frame was saved

    simulation_states: list[SimulationState] = (
        []
    )  # The measured states of the simulation

    """If we dont have cameras and the backend is not set we go to the default GLFW."""
    if len(mapping.camera_sensor.values()) == 0:
        render_backend = RenderBackend.GLFW

    """Initialize viewer object if we need to render the scene."""
    if not headless or record_settings is not None:
        match viewer_type:
            case viewer_type.CUSTOM:
                viewer = CustomMujocoViewer
            case viewer_type.NATIVE:
                viewer = NativeMujocoViewer
            case _:
                raise ValueError(
                    f"Viewer of type {viewer_type} not defined in _simulate_scene."
                )

        viewer = viewer(
            model,
            data,
            width=None if record_settings is None else record_settings.width,
            height=None if record_settings is None else record_settings.height,
            backend=render_backend,
            start_paused=start_paused,
            render_every_frame=False,
            hide_menus=(record_settings is not None),
        )

    """Record the scene if we want to record."""
    if record_settings is not None:
        if not viewer.can_record:
            raise ValueError(
                f"Selected Viewer {type(viewer).__name__} has no functionality to record."
            )
        video_step = 1 / record_settings.fps
        video_file_path = f"{record_settings.video_directory}/{scene_id}.mp4"
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        video = cv2.VideoWriter(
            video_file_path,
            fourcc,
            record_settings.fps,
            viewer.current_viewport_size(),
        )

    """
    Compute forward dynamics without actually stepping forward in time.
    This updates the data so we can read out the initial state.
    """
    mujoco.mj_forward(model, data)
    images = {
        camera_id: camera_viewer.process(model, data)
        for camera_id, camera_viewer in camera_viewers.items()
    }

    # Sample initial state.
    if sample_step is not None:
        simulation_states.append(
            SimulationStateImpl(
                data=data, abstraction_to_mujoco_mapping=mapping, camera_views=images
            )
        )

    """After rendering the initial state, we enter the rendering loop."""
    while (time := data.time) < (
        float("inf") if simulation_time is None else simulation_time
    ):

        # do control if it is time
        if time >= last_control_time + control_step:
            last_control_time = math.floor(time / control_step) * control_step

            simulation_state = SimulationStateImpl(
                data=data, abstraction_to_mujoco_mapping=mapping, camera_views=images
            )
            scene.handler.handle(simulation_state, control_interface, control_step)
            
            # Check for teleportation after handling control
            # Loop through each multi-body system (robot) in the scene
            for mbs_uuid, mbs_mujoco in mapping.multi_body_system.items():
                # Get the robot's body ID in MuJoCo
                # The id field is the base body ID of the multi-body system
                body_id = mbs_mujoco.id
                
                # Get current robot position from simulation state
                simulation_state = SimulationStateImpl(
                    data=data, abstraction_to_mujoco_mapping=mapping, camera_views=images
                )
                
                # Use the multi-body system pose to get position
                pose = simulation_state.get_multi_body_system_pose(mbs_uuid.value)
                position = Vector3(pose.position)
                
                # Check each teleport handler to see if the robot should be teleported
                for handler in teleport_handlers:
                    new_position = handler(position)
                    if new_position is not None:
                        logging.info(f"Teleporting robot from {position} to {new_position}")
                        
                        # Find the freejoint for this body in the model
                        joint_found = False
                        for i in range(model.njnt):
                            # Check if this is a free joint connected to our body
                            if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE and model.jnt_bodyid[i] == body_id:
                                joint_found = True
                                # Get the qpos address for this joint
                                qpos_addr = model.jnt_qposadr[i]
                                
                                logging.info(f"Found joint for body {body_id} at qpos index {qpos_addr}")
                                
                                # Log previous position
                                prev_x = data.qpos[qpos_addr]
                                prev_y = data.qpos[qpos_addr + 1]
                                prev_z = data.qpos[qpos_addr + 2]
                                logging.info(f"Previous qpos: ({prev_x}, {prev_y}, {prev_z})")
                                
                                # Set the position part of the free joint's qpos (first 3 values)
                                data.qpos[qpos_addr] = new_position.x  
                                data.qpos[qpos_addr + 1] = new_position.y
                                data.qpos[qpos_addr + 2] = new_position.z
                                
                                # Log new position
                                logging.info(f"Updated qpos to: ({data.qpos[qpos_addr]}, {data.qpos[qpos_addr + 1]}, {data.qpos[qpos_addr + 2]})")
                                
                                # Reset velocities for this joint
                                # First find the dof address for this joint
                                dof_addr = model.jnt_dofadr[i]
                                
                                # Set linear velocities to zero (first 3 values of qvel)
                                data.qvel[dof_addr] = 0.0
                                data.qvel[dof_addr + 1] = 0.0
                                data.qvel[dof_addr + 2] = 0.0
                                
                                # Forward to update simulation state
                                mujoco.mj_forward(model, data)
                                break
                        
                        if not joint_found:
                            logging.error(f"Failed to find free joint for body ID {body_id}")
                            # Debug info - print all joints and their body IDs
                            for i in range(model.njnt):
                                joint_type = model.jnt_type[i]
                                joint_body = model.jnt_bodyid[i]
                                logging.error(f"Joint {i}: type={joint_type}, body={joint_body}")
                                
                        break  # Only apply the first handler that returns a new position

        # sample state if it is time
        if sample_step is not None:
            if time >= last_sample_time + sample_step:
                last_sample_time = int(time / sample_step) * sample_step
                simulation_states.append(
                    SimulationStateImpl(
                        data=data,
                        abstraction_to_mujoco_mapping=mapping,
                        camera_views=images,
                    )
                )

        # step simulation
        mujoco.mj_step(model, data)
        # extract images from camera sensors.
        images = {
            camera_id: camera_viewer.process(model, data)
            for camera_id, camera_viewer in camera_viewers.items()
        }

        # render if not headless. also render when recording and if it time for a new video frame.
        if not headless or (
            record_settings is not None and time >= last_video_time + video_step
        ):
            _status = viewer.render()

            # Check if simulation was closed
            if _status == -1:
                break

        # capture video frame if it's time
        if record_settings is not None and time >= last_video_time + video_step:
            last_video_time = int(time / video_step) * video_step

            # https://github.com/deepmind/mujoco/issues/285 (see also record.cc)
            img: npt.NDArray[np.uint8] = np.empty(
                (*viewer.current_viewport_size(), 3),
                dtype=np.uint8,
            )

            mujoco.mjr_readPixels(
                rgb=img,
                depth=None,
                viewport=viewer.view_port,
                con=viewer.context,
            )
            # Flip the image and map to OpenCV colormap (BGR -> RGB)
            img = np.flipud(img)[:, :, ::-1]
            video.write(img)

    """Once simulation is done we close the potential viewer and release the potential video."""
    if not headless or record_settings is not None:
        viewer.close_viewer()

    if record_settings is not None:
        video.release()

    # Sample one final time.
    if sample_step is not None:
        simulation_states.append(
            SimulationStateImpl(
                data=data, abstraction_to_mujoco_mapping=mapping, camera_views=images
            )
        )

    logging.info(f"Scene {scene_id} done.")
    return simulation_states
