#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 09:14:53 2022

@author: savitha
"""

import random
import copy, os
from clyngor import ASP, solve

PROPERTIES = ['shape', 'color', 'material', 'siz']
domain = {}
domain['color'] = ['blue', 'brown', 'cyan', 'gray', 'green', 'purple', 'red', 'yellow', 'coral']
domain['material'] = ['rubber', 'metal']
domain['shape'] = ['cube', 'cylinder', 'sphere', 'cone']
domain['siz'] = ['large', 'small', 'medium']
region = [0,1,2,3,4,5,6,7,8]


class Template:
    def __init__(self, form, var, val_var):
        self.form = form
        self.var = var
        self.val_var = val_var
    def instantiate(self, *region):
        cal_var = []
       
        for var in self.val_var:
            func_arg = copy.deepcopy(self.val_var[var])
            if var!=0:
                for a in range(1, len(func_arg)):
                    if type(func_arg[a]) == int:
                        func_arg[a] = cal_var[func_arg[a]]
            else:
                for a in range(1, len(func_arg)):
                    if func_arg[a]=='region' and region:
                        func_arg[a] = region[0]
                   
            val = func_arg[0](*func_arg[1:])
            cal_var.append(val)
        form = self.form
        for v in range(len(self.var)):
            form = form.replace(self.var[v], str(cal_var[v]))
        return form

def generateConstraints(templates):
    #Generate 9+3 = 12 constraints
    constraints = ""
    #Generate one (within) region constraint for each of the 9 regions
    region = [0,1,2,3,4,5,6,7,8]
    for r in region:
        region_cons = random.choice([0,1,2,3])
        c = templates[region_cons].instantiate(r)
       
        c_split = c.split('.')
        for con in range(len(c_split)):
            if c_split[con]!='':
                constraints = constraints + c_split[con] + "." + "\n"
    con_across_reg = templates[4:]
    #Generate 3 across region constraints
    for i in range(3):
        n = random.choice([0,1,2,3])
        c = con_across_reg[n].instantiate()
        c_split = c.split('.')
        for con in range(len(c_split)):
            if c_split[con]!='':
                constraints = constraints + c_split[con] + "." + "\n"
    #print("Returning constraints:", constraints)
    return constraints
       

def createTemplateInstance(templates_list):

    #all objects in R' should have value V1' for P1' and value V2' for P2'
    template_1 = templates_list[0]
    vars1 = ["R'", "P1'", "V1'", "P2'", "V2'"]
    val_var = {}
    val_var[0] = [lambda region : region, 'region']
    val_var[1] = [lambda PROPERTIES : random.choice(PROPERTIES), PROPERTIES]
    val_var[2] = [lambda x : random.choice(domain[x]), 1]
    val_var[3] = [lambda y, PROPERTIES: random.choice(list(filter(lambda x: (x != y), PROPERTIES))), 1, PROPERTIES]
    val_var[4] = [lambda x: random.choice(domain[x]), 3]

    t1 = Template(template_1, vars1, val_var)
    #t1_instance = t1.instantiate()
    #print(t1_instance)

    #all objects in R' should have value V1' for P1' or value V2' for P2'
    template_2 = templates_list[1]
    vars1 = ["R'", "P1'", "V1'", "P2'", "V2'"]
    val_var = {}
    val_var[0] = [lambda region : region, 'region']
    val_var[1] = [lambda PROPERTIES : random.choice(PROPERTIES), PROPERTIES]
    val_var[2] = [lambda x : random.choice(domain[x]), 1]
    val_var[3] = [lambda y, PROPERTIES: random.choice(list(filter(lambda x: (x != y), PROPERTIES))), 1, PROPERTIES]
    val_var[4] = [lambda x: random.choice(domain[x]), 3]
    t2 = Template(template_2, vars1, val_var)
    #t2_instance = t2.instantiate()
    #print(t2_instance)

    #all objects in R' should not have value V1' for P1'
    template_3 = templates_list[2]
    vars1 = ["R'", "P1'", "V1'"]
    val_var = {}
    val_var[0] = [lambda region : region, 'region']
    val_var[1] = [lambda PROPERTIES : random.choice(PROPERTIES), PROPERTIES]
    val_var[2] = [lambda x : random.choice(domain[x]), 1]
    t3 = Template(template_3, vars1, val_var)
    #t3_instance = t3.instantiate()
    #print(t3_instance)

    #atleast one object of same value for a property P in two regions R1' and R2'
    template_4 = templates_list[3]
    vars1 = ["R1'", "R2'",  "P1'", "N'"]
    val_var = {}
    val_var[0] = [lambda region : random.choice(region), region]
    val_var[1] = [lambda y, region: random.choice(list(filter(lambda x: (x != y), region))), 0, region]
    val_var[2] = [lambda PROPERTIES : random.choice(PROPERTIES), PROPERTIES]
    val_var[3] = [lambda x: random.choice(x), [1,2]]
    t4 = Template(template_4, vars1, val_var)
    #t4_instance = t4.instantiate()
    #print(t4_instance)

    #atleast 1 object in two regions R1' and R2' have same value for property P1' for two objects with same value for property P2'
    template_5 = templates_list[4]
    vars1 = ["R1'", "R2'",  "P1'", "P2'", "V2'", "N'"]
    val_var = {}
    val_var[0] = [lambda region : random.choice(region), region]
    val_var[1] = [lambda y, region: random.choice(list(filter(lambda x: (x != y), region))), 0, region]
    val_var[2] = [lambda PROPERTIES : random.choice(PROPERTIES), PROPERTIES]
    val_var[3] = [lambda y, PROPERTIES: random.choice(list(filter(lambda x: (x != y), PROPERTIES))), 2, PROPERTIES]
    val_var[4] = [lambda x : random.choice(domain[x]), 3]
    val_var[5] = [lambda x: random.choice(x), [1,2]]
    t5 = Template(template_5, vars1, val_var)
    #t5_instance = t5.instantiate()
    #print(t5_instance)


    #cannot have any object of same value for a property P in two regions R1' and R2'
    template_6 = templates_list[5]
    vars1 = ["R1'", "R2'",  "P1'", "N'"]
    val_var = {}
    val_var[0] = [lambda region : random.choice(region), region]
    val_var[1] = [lambda y, region: random.choice(list(filter(lambda x: (x != y), region))), 0, region]
    val_var[2] = [lambda PROPERTIES : random.choice(PROPERTIES), PROPERTIES]
    val_var[3] = [lambda x: random.choice(x), [1, 2]]
    t6 = Template(template_6, vars1, val_var)
    t6_instance1 = t6.instantiate()
    #t6_instance2 = t6.instantiate()
    #print(t6_instance1)

    #two regions R1' and R2' cannot have same value for property P1' for N objects same value for property P2'
    template_7 = templates_list[6]
    vars1 = ["R1'", "R2'",  "P1'", "P2'", "V2'", "N'"]
    val_var = {}
    val_var[0] = [lambda region : random.choice(region), region]
    val_var[1] = [lambda y, region: random.choice(list(filter(lambda x: (x != y), region))), 0, region]
    val_var[2] = [lambda PROPERTIES : random.choice(PROPERTIES), PROPERTIES]
    val_var[3] = [lambda y, PROPERTIES: random.choice(list(filter(lambda x: (x != y), PROPERTIES))), 2, PROPERTIES]
    val_var[4] = [lambda x : random.choice(domain[x]), 3]
    val_var[5] = [lambda x: random.choice(x), [1, 2]]
    t7 = Template(template_7, vars1, val_var)
    #t7_instance = t7.instantiate()
    #print(t7_instance)

    #there are exactly two objects with P1' = V1' in R1', n=2.
    template_8 = templates_list[7]
    vars1 = ["R1'", "P1'", "V1'", "N'"]
    val_var = {}
    val_var[0] = [lambda region : region, 'region']
    val_var[1] = [lambda PROPERTIES : random.choice(PROPERTIES), PROPERTIES]
    val_var[2] = [lambda x : random.choice(domain[x]), 1]
    val_var[3] = [lambda x: random.choice(x), [1,2]]
    t8 = Template(template_8, vars1, val_var)
    #t8_instance = t8.instantiate()
    #print(t8_instance)
    templates = [t1, t2, t3, t8, t4, t5, t6, t7]
    return templates

def generateEnvironment(args, num_objects, env_id, environment_constraints_dir):
    templates_list=[]
    file1 = open(args.constraint_template_path, 'r')
    Lines = file1.readlines()
    for line in Lines:
        template = line.split('=')[1]
        templates_list.append(template)
    templates = createTemplateInstance(templates_list)
    file1 = open(args.general_constraints_path, 'r')
    Lines = file1.readlines()
    background = ""
    for line in Lines:
        background = background+line

    background = background+"\n"+"object(0.."+str(num_objects)+")."+"\n"    
    satisfiable = False
    while(not(satisfiable)):
        asp_file = open(os.path.join(environment_constraints_dir, str(env_id)+".lp"), "w")
        constraints = generateConstraints(templates)
        asp_code = background+constraints+"\n"+"#show hasProperty/3. #show at/2."
        n1 = asp_file.write(asp_code)
        asp_file.close()
        answers = ASP(asp_code)
        count = 0
        for answer in answers:
            count = count+1
            if(count>=1):
                satisfiable = True
                print("Satisfiable")
                break
    
    




    