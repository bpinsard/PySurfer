#!/usr/bin/env python
"""
an object to view ROIs data
"""
# Author: Basile Pinsard, Gael Varoquaux
# License: BSD
import numpy as np
import os

from traits.api import HasTraits, Instance, Array, Int, Float, \
    Bool, Dict, on_trait_change, Range, Property, Button
from traitsui.api import View, Item, HGroup, Group

from tvtk.api import tvtk
from tvtk.pyface.scene import Scene

from mayavi import mlab
from mayavi.core.api import PipelineBase, Source
from mayavi.core.ui.api import SceneEditor, MlabSceneModel

class RoisSlicer(HasTraits):
    # The data to plot
    data = Array

    index = Range(value=0, low='_zero',
                  high='_volume_max', mode='spinner',
                  help="Number of the volume to display",)
    _zero = Int(0)
    _volume_max = Property(depends_on="rois_data")

    rois_data = Dict(Int,Array)

    data_range_low = Float(-3)
    data_range_high = Float(3)

    def _mayavi_dialog_fired(self):
        mlab.show_pipeline()

    def _get__volume_max(self):
        if len(self.rois_data)>0:
            return (self.rois_data.values()[0].shape+(1,))[1]-1
        return 0
    

    # The position of the view
    position = Array(shape=(3,))

    # The 4 views displayed
    scene3d = Instance(MlabSceneModel, ())
    scene_x = Instance(MlabSceneModel, ())
    scene_y = Instance(MlabSceneModel, ())
    scene_z = Instance(MlabSceneModel, ())

    # The data source
    data_src = Instance(Source)

    # The image plane widgets of the 3D scene
    ipw_3d_x = Instance(PipelineBase)
    ipw_3d_y = Instance(PipelineBase)
    ipw_3d_z = Instance(PipelineBase)

    # The cursors on each view:
    cursors = Dict()

    disable_render = Bool

    _axis_names = dict(x=0, y=1, z=2)

    #------------------------------------------------------------------------
    # Object interface
    #------------------------------------------------------------------------
    def __init__(self, rois_mask, rois_mask_highres=None, 
                 rois_labels=None,lut_path=None,**traits):

        self.disable_render = True

        self._rois_mask = rois_mask
        self._rois_data = rois_mask.get_data()
        self._rois_mask_highres = rois_mask_highres
        if self._rois_mask_highres is None:
            self._rois_mask_highres = self._rois_mask

        self._rois_labels = rois_labels
        if self._rois_labels is None:
            self._rois_labels = np.unique(self._rois_data[self._rois_data>0])
        self._rois_mask_data = np.zeros_like(self._rois_data, dtype=np.float)
        for l in self._rois_labels:
            self._rois_mask_data[self._rois_data==l] = l

        aw = np.argwhere(self._rois_mask_data>0)
        awmin,awmax = aw.min(0)-1,aw.max(0)+1
        print awmin, awmax
        self._bbox = [slice(l,t) for l,t in zip(awmin,awmax)]
        del aw

        
        self._spacing = self._rois_mask.get_header().get_zooms() 
        self._origin = awmin*self._spacing

        if lut_path is None:
            lut_path = os.path.join(os.environ['FREESURFER_HOME'],
                                    'FreeSurferColorLUT.txt')
        lut_file = open(lut_path)
        self._lut = dict()
        for l in lut_file.readlines():
            if len(l)>4 and l[0]!='#':
                l = l.split()
                self._lut[int(l[0])] = (l[1],
                                        tuple(float(c)/255. for c in l[2:5]))
        lut_file.close()

        print self._bbox
        self._rois_mask_data[self._rois_mask_data==0] = np.nan
        traits['data'] = self._rois_mask_data[self._bbox]

        super(RoisSlicer, self).__init__(**traits)
        # Force the creation of the image_plane_widgets:
        print 'create plane widgets'
        self.ipw_3d_x
        self.ipw_3d_y
        self.ipw_3d_z

        lm = self.ipw_3d_x.module_manager.scalar_lut_manager
        lm.data_range = (
            self.data_range_low, self.data_range_high)
        lm.lut.nan_color = (0,0,0,0)
        lm.use_default_range = False

        surf_spacing = self._rois_mask_highres.get_header().get_zooms()
        print 'create surfaces'
        self._surfaces = []
        if self._rois_mask == self._rois_mask_highres:
            hr_mask = self._rois_mask_data
        else:
            hr_mask = self._rois_mask_highres.get_data()
        for l in self._rois_labels:
            mask = (hr_mask == l)
            aw = np.argwhere(mask)
            awmin,awmax = aw.min(0),aw.max(0)
            bbox = [slice(b,t) for b,t in zip(awmin,awmax)]
            del aw
            src = mlab.pipeline.scalar_field(mask[bbox].astype(np.uint8))
            surf = mlab.pipeline.iso_surface(src,
                                             figure=self.scene3d.mayavi_scene,
                                             contours = [1], opacity=0.3)
            surf.actor.property.color = self._lut[l][1]
            surf.actor.mapper.scalar_visibility = False
            surf.actor.actor.position = awmin * surf_spacing - \
                [s.start*sp for s,sp in zip(self._bbox,self._spacing)]
            surf.actor.actor.scale = surf_spacing
            self._surfaces.append(surf)
            del mask

        self.disable_render = False



    #------------------------------------------------------------------------
    # Default values
    #------------------------------------------------------------------------
    def _position_default(self):
        return 0.5*np.array(self.data.shape)

    def _data_src_default(self):
        sf = mlab.pipeline.scalar_field(self.data,
                                        figure=self.scene3d.mayavi_scene,
                                        name='Data',)
        print self._origin, self._spacing
        #sf.origin = self._origin
        sf.spacing = self._spacing
        return sf

    def make_ipw_3d(self, axis_name):
        ipw = mlab.pipeline.image_plane_widget(
            self.data_src,
            figure=self.scene3d.mayavi_scene,
            plane_orientation='%s_axes' % axis_name,
            name='Cut %s' % axis_name)
        ipw.widgets[0].texture_interpolate = False
        ipw.widgets[0].reslice_interpolate = 'nearest_neighbour'
        return ipw

    def _ipw_3d_x_default(self):
        return self.make_ipw_3d('x')

    def _ipw_3d_y_default(self):
        return self.make_ipw_3d('y')

    def _ipw_3d_z_default(self):
        return self.make_ipw_3d('z')

    #------------------------------------------------------------------------
    # Scene activation callbacks
    #------------------------------------------------------------------------
    @on_trait_change('scene3d.activated')
    def display_scene3d(self):
        outline = mlab.pipeline.outline(
            self.data_src,
            figure=self.scene3d.mayavi_scene,)
        self.scene3d.mlab.view(40, 50)
        # Interaction properties can only be changed after the scene
        # has been created, and thus the interactor exists
        for ipw in (self.ipw_3d_x, self.ipw_3d_y, self.ipw_3d_z):
            ipw.ipw.interaction = 0
        self.scene3d.scene.background = (0, 0, 0)
        # Keep the view always pointing up
        self.scene3d.scene.interactor.interactor_style = \
                                 tvtk.InteractorStyleTerrain()
        self.update_position()


    def make_side_view(self, axis_name):
        scene = getattr(self, 'scene_%s' % axis_name)
        scene.scene.parallel_projection = True
        ipw_3d   = getattr(self, 'ipw_3d_%s' % axis_name)

        # We create the image_plane_widgets in the side view using a
        # VTK dataset pointing to the data on the corresponding
        # image_plane_widget in the 3D view (it is returned by
        # ipw_3d._get_reslice_output())
        side_src = ipw_3d.ipw._get_reslice_output()
        ipw = mlab.pipeline.image_plane_widget(
                            side_src,
                            plane_orientation='z_axes',
                            vmin=self.data.min(),
                            vmax=self.data.max(),
                            figure=scene.mayavi_scene,
                            name='Cut view %s' % axis_name,
                            )
        ipw.widgets[0].texture_interpolate = False
        ipw.widgets[0].reslice_interpolate = 'nearest_neighbour'
        ipw.module_manager.scalar_lut_manager.data_range = (
            self.data_range_low, self.data_range_high)


        lut_manager= ipw.module_manager.scalar_lut_manager
        lut_manager.lut.nan_color = (0,0,0,0)
        lut_manager.use_default_range = False

        setattr(self, 'ipw_%s' % axis_name, ipw)

        # Extract the spacing of the side_src to convert coordinates
        # into indices
        spacing = side_src.spacing

        # Make left-clicking create a crosshair
        ipw.ipw.left_button_action = 0

        x, y, z = self.position
        cursor = mlab.points3d(x, y, z,
                            mode='axes',
                            color=(0, 0, 0),
                            scale_factor=2*max(self.data.shape),
                            figure=scene.mayavi_scene,
                            name='Cursor view %s' % axis_name,
                        )
        self.cursors[axis_name] = cursor

        # Add a callback on the image plane widget interaction to
        # move the others
        this_axis_number = self._axis_names[axis_name]
        def move_view(obj, evt):
            # Disable rendering on all scene
            position = list(obj.GetCurrentCursorPosition()*spacing)[:2]
            position.insert(this_axis_number, self.position[this_axis_number])
            # We need to special case y, as the view has been rotated.
            if axis_name is 'y':
                position = position[::-1]
            self.position = position

        ipw.ipw.add_observer('InteractionEvent', move_view)
        ipw.ipw.add_observer('StartInteractionEvent', move_view)

        # Center the image plane widget
        ipw.ipw.slice_position = 0.5*self.data.shape[
                                        self._axis_names[axis_name]]

        # 2D interaction: only pan and zoom
        scene.scene.interactor.interactor_style = \
                                 tvtk.InteractorStyleImage()
        scene.scene.background = (0, 0, 0)

        # Some text:
        mlab.text(0.01, 0.8, axis_name, width=0.08)

        # Choose a view that makes sens
        views = dict(x=(0, 0), y=(90, 180), z=(0, 0))
        mlab.view(views[axis_name][0],
                  views[axis_name][1],
                  focalpoint=0.5*np.array(self.data.shape),
                  figure=scene.mayavi_scene)
        scene.scene.camera.parallel_scale = 0.52*np.mean(self.data.shape)

    @on_trait_change('scene_x.activated')
    def display_scene_x(self):
        return self.make_side_view('x')

    @on_trait_change('scene_y.activated')
    def display_scene_y(self):
        return self.make_side_view('y')

    @on_trait_change('scene_z.activated')
    def display_scene_z(self):
        return self.make_side_view('z')


    #---------------------------------------------------------------------------
    # Traits callback
    #---------------------------------------------------------------------------
    @on_trait_change('position')
    def update_position(self):
        """ Update the position of the cursors on each side view, as well
            as the image_plane_widgets in the 3D view.
        """
        # First disable rendering in all scenes to avoid unecessary
        # renderings
        self.disable_render = True

        # For each axis, move image_plane_widget and the cursor in the
        # side view
        for axis_name, axis_number in self._axis_names.iteritems():
            ipw3d = getattr(self, 'ipw_3d_%s' % axis_name)
            ipw3d.ipw.slice_position = self.position[axis_number]

            # Go from the 3D position, to the 2D coordinates in the
            # side view
            position2d = list(self.position)
            position2d.pop(axis_number)
            if axis_name is 'y':
                position2d = position2d[::-1]
            # Move the cursor
            # For the following to work, you need Mayavi 3.4.0, if you
            # have a less recent version, use 'x=[position2d[0]]'
            self.cursors[axis_name].mlab_source.set(
                                                x=position2d[0],
                                                y=position2d[1],
                                                z=0)

        # Finally re-enable rendering
        self.disable_render = False

    @on_trait_change('disable_render')
    def _render_enable(self):
        for scene in (self.scene3d, self.scene_x, self.scene_y,
                                                  self.scene_z):
            scene.scene.disable_render = self.disable_render

    @on_trait_change('data')
    def _redraw(self):
        self.data_src.mlab_source.scalars = self.data

    @on_trait_change('[data_range_low,data_range_high]')
    def set_colormap(self):
        self.disable_render = True
        for ipw in [self.ipw_3d_x, self.ipw_x, self.ipw_y, self.ipw_z]:
            lm = ipw.module_manager.scalar_lut_manager
            lm.data_range = (
                self.data_range_low, self.data_range_high)
            lm.lut.nan_color = (0,0,0,0)
            lm.use_default_range = False

        self.disable_render = False

    #---------------------------------------------------------------------------
    # The layout of the dialog created
    #---------------------------------------------------------------------------

    mayavi_dialog = Button('Show pipeline')

    def _mayavi_dialog_fired(self):
        mlab.show_pipeline()

    view = View(
        Group(
            HGroup('index',
                   'data_range_low','data_range_high',
                   'mayavi_dialog'),            
        HGroup(
                Group(
                    Item('scene_y',
                         editor=SceneEditor(scene_class=Scene),
                         height=250, width=300),
                    Item('scene_z',
                         editor=SceneEditor(scene_class=Scene),
                         height=250, width=300),
                    show_labels=False,
                    ),
                Group(
                    Item('scene_x',
                         editor=SceneEditor(scene_class=Scene),
                         height=250, width=300),
                    Item('scene3d',
                         editor=SceneEditor(scene_class=Scene),
                         height=250, width=300),
                    show_labels=False,
                    ),
                )),
        resizable=True,
        title='ROIs Slicer',
        )

    @on_trait_change('[rois_data,index]')
    def set_rois_data(self):
        if self.index >= self._volume_max:
            self.index=0
        tmp = np.empty(self.data.shape)
        tmp.fill(np.nan)
        for i, d in self.rois_data.items():
            plot_mask = self._rois_mask_data[self._bbox] == i
            if np.count_nonzero(plot_mask) != d.shape[0] and d.shape[0]>1:
                raise ValueError(
                    'data should have the same number of voxels as roi')
            if d.ndim > 1:
                plot_data = d[:, self.index]
            else:
                plot_data = d
            tmp[plot_mask] = plot_data

        self.data = tmp
