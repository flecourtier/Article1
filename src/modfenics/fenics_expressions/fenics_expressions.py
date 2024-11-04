# ###########
# # Imports #
# ###########

# from dolfin import *
import dolfin as dol
from dolfin.function.expression import (
    BaseExpression,
    _select_element,
    _InterfaceExpression,
)

# ####################
# # MyUserExpression #
# ####################

# class MyUserExpression(BaseExpression):
#     """JIT Expressions"""

#     def __init__(self, degree, domain):
#         cell = domain.ufl_cell()
#         element = _select_element(
#             family=None, cell=cell, degree=degree, value_shape=()
#         )

#         self._cpp_object = _InterfaceExpression(self, ())

#         BaseExpression.__init__(
#             self,
#             cell=cell,
#             element=element,
#             domain=domain,
#             name=None,
#             label=None,
#         )
        
# MyUserExpression for 2x2 matrices
class MyUserExpression22(BaseExpression):
    """JIT Expressions"""

    def __init__(self, degree, domain):
        cell = domain.ufl_cell()
        element = _select_element(
            family=None, cell=cell, degree=degree, value_shape=(2,2)
        )

        self._cpp_object = _InterfaceExpression(self, (2,2))

        BaseExpression.__init__(
            self,
            cell=cell,
            element=element,
            domain=domain,
            name=None,
            label=None,
        )

# ######################
# # FENICS Expressions #
# ######################

# class UexExpr(MyUserExpression):
#     def __init__(self, params, degree, domain, pb_considered):
#         super().__init__(degree, domain)
#         self.mu = params
#         self.pb_considered = pb_considered
    
#     def eval(self, value, x):
#         value[0] = self.pb_considered.u_ex(dol, x, self.mu)

# class FExpr(MyUserExpression):
#     def __init__(self, params, degree, domain, pb_considered):
#         super().__init__(degree, domain)
#         self.mu = params
#         self.pb_considered = pb_considered

#     def eval(self, value, x):
#         value[0] = self.pb_considered.f(dol, x, self.mu)
        
# class GExpr(MyUserExpression): # dirichlet
#     def __init__(self, params, degree, domain, pb_considered):
#         super().__init__(degree, domain)
#         self.mu = params
#         self.pb_considered = pb_considered
    
#     def eval(self, value, x):
#         value[0] = self.pb_considered.g(dol, x, self.mu)
        
# class GNExpr(MyUserExpression): # neumann
#     def __init__(self, params, degree, domain, pb_considered):
#         super().__init__(degree, domain)
#         self.mu = params
#         self.pb_considered = pb_considered
    
#     def eval(self, value, x):
#         value[0] = self.pb_considered.gn(dol, x, self.mu)
        
# class GRExpr(MyUserExpression): # robin
#     def __init__(self, params, degree, domain, pb_considered):
#         super().__init__(degree, domain)
#         self.mu = params
#         self.pb_considered = pb_considered
    
#     def eval(self, value, x):
#         value[0] = self.pb_considered.gr(dol, x, self.mu)

# # Temporary (TestCase 6 + 7)
# class HExtExpr(MyUserExpression):
#     def __init__(self, params, degree, domain, pb_considered):
#         super().__init__(degree, domain)
#         self.mu = params
#         self.pb_considered = pb_considered
    
#     def eval(self, value, x):
#         value[0] = self.pb_considered.h_ext(dol, x, self.mu)

# # Temporary (TestCase 6 + 7)   
# class HIntExpr(MyUserExpression):
#     def __init__(self, params, degree, domain, pb_considered):
#         super().__init__(degree, domain)
#         self.mu = params
#         self.pb_considered = pb_considered
    
#     def eval(self, value, x):
#         value[0] = self.pb_considered.h_int(dol, x, self.mu)
        
# Use only in the 3rd TestCase
class AnisotropyExpr(MyUserExpression22):
    def __init__(self, params, degree, domain, pb_considered):
        super().__init__(degree, domain)
        self.mu = params
        self.pb_considered = pb_considered

    def eval(self, value, x):
        val = self.pb_considered.anisotropy_matrix(dol, x, self.mu)
        value[0] = val[0]
        value[1] = val[1]
        value[2] = val[2]
        value[3] = val[3]

import sympy as sp
import dolfin as df

def get_expr_from_sympy(params, degree, domain, fct):
    nb_parameters = len(params)
    params_dict = {f'p{i+1}': params[i] for i in range(nb_parameters)}
    
    x, y = sp.symbols('xx yy')
    xy = [x, y]
    mu = sp.symbols(' '.join(params_dict.keys()))
    
    fct_sympy = fct(sp, xy, mu)
    
    # Crée le dictionnaire des paramètres pour df.Expression
    expression_params = {str(symbol): value for symbol, value in zip(mu, params_dict.values())}
    
    # Remplace 'xx' par 'x[0]' et 'yy' par 'x[1]' pour l'expression dans df.Expression
    fct_fe = df.Expression(sp.ccode(fct_sympy).replace('xx', 'x[0]').replace('yy', 'x[1]'),degree=degree, domain=domain, **expression_params)
    
    return fct_fe

def get_uex_expr(params, degree, domain, pb_considered):
    return get_expr_from_sympy(params, degree, domain, pb_considered.u_ex)

def get_f_expr(params, degree, domain, pb_considered):
    return get_expr_from_sympy(params, degree, domain, pb_considered.f)

def get_g_expr(params, degree, domain, pb_considered):
    return get_expr_from_sympy(params, degree, domain, pb_considered.g)

def get_gn_expr(params, degree, domain, pb_considered):
    return get_expr_from_sympy(params, degree, domain, pb_considered.gn)

def get_gr_expr(params, degree, domain, pb_considered):
    return get_expr_from_sympy(params, degree, domain, pb_considered.gr)

def get_h_ext_expr(params, degree, domain, pb_considered):
    return get_expr_from_sympy(params, degree, domain, pb_considered.h_ext)

def get_h_int_expr(params, degree, domain, pb_considered):
    return get_expr_from_sympy(params, degree, domain, pb_considered.h_int)

# def get_anisotropy_expr(params, degree, domain, pb_considered):
#     return get_expr_from_sympy(params, degree, domain, pb_considered.anisotropy_matrix)

