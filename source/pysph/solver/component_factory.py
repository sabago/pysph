"""
Factory to create any component.
"""

import logging
logger = logging.getLogger()

class ComponentFactory:
    """
    Factory class to create any component.
    """
    def __init__(self):
        """
        """
        raise SystemError, 'Do not Instantiate the ComponentFactory class'

    @staticmethod
    def get_component(comp_category, comp_type, *args, **kwargs):
        """
        Creates and returns the requested component type.

        **Parameters**
            
            - comp_category - category to which the component belogns.
            - comp_name - string identifying the type of component needed.
            - *args, **kwargs - any arguements to be passed to the constructor
              of the requested component.

        """
        if comp_category == 'ode_stepper':
            return ComponentFactory.get_ode_stepper(comp_type, *args, **kwargs)
        elif comp_category == 'integrator':
            return ComponentFactory.get_integrator(comp_type, *args, **kwargs)
        elif comp_category == 'copiers':
            return ComponentFactory.get_copier(comp_type, *args, **kwargs)
        elif comp_category == 'dummies':
            return ComponentFactory.get_dummy_component(comp_type, *args,
                                                        **kwargs)
        else:        
            logger.warn('Cannot produce %s'%(comp_type))
            return None

    @staticmethod
    def get_ode_stepper(comp_type, *args, **kwargs):
        """
        Creates and returns the requested ode stepper.
        """
        import pysph.solver.integrator_base
        import pysph.solver.dummy_components
        import pysph.solver.xsph_component
        import pysph.solver.runge_kutta_integrator

        if comp_type == 'base' or comp_type == 'euler':
            return pysph.solver.integrator_base.ODEStepper(*args, **kwargs)
        elif comp_type == 'ya_stepper':
            return pysph.solver.dummy_components.YAStepper(*args, **kwargs)
        elif comp_type == 'euler_xsph_position_stepper':
            return pysph.solver.xsph_component.EulerXSPHPositionStepper(
                *args, **kwargs)
        elif comp_type == 'rk2_second_step':
            return pysph.solver.runge_kutta_integrator.RK2SecondStep(
                *args, **kwargs)
        elif comp_type == 'rk2_xsph_step1_position_stepper':
            return pysph.solver.xsph_component.RK2Step1XSPHPositionStepper(
                *args, **kwargs)
        elif comp_type == 'rk2_xsph_step2_position_stepper':
            return pysph.solver.xsph_component.RK2Step2XSPHPositionStepper(
                *args, **kwargs)
        else:
            logger.warn('Cannot produce %s'%(comp_type))
            return None
        
    @staticmethod
    def get_integrator(comp_type, *args, **kwargs):
        """
        Creates and returns the requested integrator.
        """
        import pysph.solver.integrator_base

        if comp_type == 'base' or comp_type =='euler':
            return pysph.solver.integrator_base.Integrator(*args, **kwargs)
        else:
            logger.warn('Cannot produce %s'%(comp_type))
            return None

    @staticmethod
    def get_copier(comp_type, *args, **kwargs):
        """
        Creates and returns the requested copier.
        """
        import pysph.solver.array_copier

        if comp_type == 'copier':
            return pysph.solver.array_copier.ArrayCopier(*args, **kwargs)
        else:
            logger.warn('Cannot produce %s'%(comp_type))
            return None

    @staticmethod
    def get_dummy_component(comp_type, *args, **kwargs):
        """
        Creates and returns one of the dummy components. Typically these
        components are used for tests etc.
        """
        import pysph.solver.dummy_components

        if comp_type == 'ya_stepper':
            return pysph.solver.dummy_components.YAStepper(args, kwargs)
        else:
            logger.warn('Cannot produce %s'%(comp_type))
            return None
