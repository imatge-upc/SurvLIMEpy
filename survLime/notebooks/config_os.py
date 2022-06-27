import numpy as np

from lab_names import lab_name_dict

rename_lab_values = {
    'lab_dict': lab_name_dict
}

custom_missing_indicator = {'cols': ['braf_mutation', 'cdkn2a_mutation', 'amount_sun_exposure']}

transform_to_numeric = {'cols': ['cutaneous_biopsy_breslow']}
transform_to_datetime = {'cols': ['cb_examined_at', 'birthdate']}
transform_to_object = {'cols': ['high_abv_frequency']}

drop_na_cols = {
    'cols': ['cutaneous_biopsy_breslow', 'cutaneous_biopsy_ulceration']}
drop_zero_cols = {'cols': ['cutaneous_biopsy_breslow']}

get_institutions = {'institutions': [1]}

compute_age = {'diag_time': 'cb_examined_at', 'birthdate': 'birthdate'}

tumour_location_transform_variables = {'acral': ['nail_toe', 'toe_left', 'toe_right', 'foot_right_plantar', 'finger_left', 'foot_left_dorsal', 'foot_left_plantar', 'hand_palmar', 'foot_dorsal', 'finger_right', 'foot_right_dorsal', 'foot_plantar', 'nail_finger', 'nail_finger_left', 'nail_finger_right', 'nail_toe_left', 'finger', 'hand', 'nail_toe_right', 'plantar', 'hand_right_anterior', 'hand_left_anterior'],
                                       'head and neck': ['head', 'cheek', 'face', 'cheek_right', 'forehead', 'eye_left', 'scalp', 'ear_right', 'nose', 'eyelid', 'ear', 'ear_left', 'laterocervical_left', 'cheek_left', 'neck', 'chin', 'laterocervical_right', 'lip', 'eyelid_right', 'eye_right', 'eye', 'eyelid_left'],
                                       'lower limbs': ['leg_right_anterior', 'thigh_posterior', 'thigh_left', 'buttock_right', 'leg_right', 'buttock_left', 'thigh_anterior', 'foot_right', 'leg_left', 'thigh_right', 'leg', 'foot_left', 'leg_anterior', 'thigh', 'popliteal_left', 'leg_left_posterior', 'leg_left_anterior', 'leg_posterior', 'inguinal_left', 'thigh_left_posterior', 'toe', 'thigh_right_anterior', 'thigh_left_anterior', 'anus', 'buttock', 'leg_right_posterior', 'foot', 'thigh_right_posterior', 'popliteal_right', 'inguinal_right'],
                                       'upper limbs': ['arm_right', 'arm_anterior', 'forearm_left', 'arm_left_anterior', 'forearm_right', 'arm_left', 'axilla_right', 'shoulder_left', 'shoulder_right', 'axilla_left', 'forearm_anterior', 'arm_right_anterior', 'arm', 'forearm', 'hand_anterior', 'hand_right', 'forearm_right_anterior', 'arm_left_posterior', 'forearm_left_posterior', 'arm_right_posterior', 'forearm_left_anterior', 'hand_left', 'arm_posterior', 'shoulder', 'forearm_posterior', 'forearm_right_posterior', 'hand_right_posterior'],
                                       'mucosa': ['vulva', 'vagina', 'nasopharynx', 'oropharynx'],
                                       'other': ['penis_scrotum', 'other', 'gastrointestinal'],
                                       'trunk': ['back', 'abdomen', 'lumbar', 'back_middle', 'back_left', 'chest', 'chest_right', 'back_right', 'trunk', 'chest_left', 'abdomen_right', 'abdomen_left', 'trunk_anterior', 'abdomen_middle', 'trunk_posterior', 'chest_middle', 'iliac_right', 'iliac_left']
                                       }


link_tumour_part_to_parent = {'tumour_location_transform_variables': tumour_location_transform_variables,
                              'col': 'primary_tumour_location_coded'
                              }

cutaneous_biopsy_regression_transform_dict = {'absent': ['none'],
                                              'partial': ['lt_25', 'lt_50', '25_50'],
                                              'extensive': ['gt_50', '50_75', 'gt_75']}

transform_cb_regression = {'cutaneous_biopsy_regression_transform_dict': cutaneous_biopsy_regression_transform_dict,
                           'col': 'cutaneous_biopsy_regression'
                           }

categorical_encoder = {'categorical_variables': {"primary_tumour_location_coded": ["trunk",
                                                                                   "acral",
                                                                                   "head and neck",
                                                                                   "lower limbs",
                                                                                   "upper limbs",
                                                                                   "mucosa",
                                                                                   ],

                                                 "cutaneous_biopsy_predominant_cell_type": ["epitheloid",
                                                                                            "fusocellular",
                                                                                            "pleomorphic",
                                                                                            "sarcomatoid",
                                                                                            "small_cell",
                                                                                            "spindle"],

                                                 "cutaneous_biopsy_histological_subtype": ["superficial_spreading",
                                                                                           "acral_lentiginous",
                                                                                           "desmoplastic",
                                                                                           "lentiginous_malignant",
                                                                                           "mucosal",
                                                                                           "nevoid",
                                                                                           "nodular",
                                                                                           "spitzoid",
                                                                                           ],

                                                 "patient_hair_color": ["black",
                                                                        "blond",
                                                                        "brown",
                                                                        "red"]},
                       'categorical_variables_references': {'primary_tumour_location_coded': 'trunk',
                                                            'cutaneous_biopsy_predominant_cell_type': 'epitheloid',
                                                            'cutaneous_biopsy_histological_subtype': 'superficial_spreading',
                                                            'patient_hair_color': 'brown'
                                                            }
                       }


exponential_transformer = {'exponential_variables': ["cutaneous_biopsy_breslow",
                                                     "cutaneous_biopsy_mitotic_index"]
                           }

gender_encoder = {'gender_variables': ["patient_gender"],
                  'gender_dictionary': {"male": 0,
                                        "female": 1}}

absent_present_encoder = {'absent_present_variables': ["cutaneous_biopsy_ulceration",
                                                       "cutaneous_biopsy_satellitosis",
                                                       "cutaneous_biopsy_vascular_invasion",
                                                       "cutaneous_biopsy_neurotropism",
                                                       "cutaneous_biopsy_lymphatic_invasion"],
                          'absent_dictionary': {"absent": 0,
                                                "present": 1}
                          }

convert_categories_to_nan = {'data_to_convert_to_na': {'primary_tumour_location_coded': 'other',
                                                       'cutaneous_biopsy_histological_subtype': 'other',
                                                       'cutaneous_biopsy_predominant_cell_type': 'other',
                                                       'patient_hair_color': 'other',
                                                       'patient_eye_color': 'other',
                                                       }
                             }

lab_encoder = {'lab_transforms': {'LAB1300': [4,
                                              11],

                                  'LAB1301': [130,
                                              400],

                                  'LAB1307': [2.5,
                                              7],

                                  'LAB1309': [0.9,
                                              4.5],

                                  'LAB1311': [0.1,
                                              1],

                                  'LAB1313': [0,
                                              0.5],

                                  'LAB1314': [120,
                                              170],

                                  'LAB1316': [0,
                                              0.2],

                                  'LAB2404': [5,
                                              40],

                                  'LAB2405': [5,
                                              40],

                                  'LAB2406': [5,
                                              40],

                                  'LAB2407': [0.2,
                                              1.2],

                                  'LAB2419': [250,
                                              450],

                                  'LAB2422': [65,
                                              110],

                                  'LAB2467': [0.3,
                                              1.3],

                                  'LAB2476': [50,
                                              150],

                                  'LAB2498': [63,
                                              80],

                                  'LAB2544': [0.1,
                                              2.3],

                                  'LAB2679': [0,
                                              0.2],

                                  'LAB4176': [0,
                                              12],

                                  'LAB2469': [0,
                                              160]
                                  }
               }


################################################


ordinal_encoder = {'ordinal_variables': ["cutaneous_biopsy_regression",
                                         "patient_eye_color",
                                         "high_abv_frequency",
                                         'nevi_count',
                                         'amount_sun_exposure',
                                         'nca',
                                         'braf_mutation',
                                         'cdkn2a_mutation',
                                         'cutaneous_biopsy_associated_nevus'
                                         ],
                   'transforms': {'cutaneous_biopsy_regression_transform': {'absent': 1,
                                                                            'partial': 2,
                                                                            'extensive': 3},
                                  'patient_eye_color_transform': {"blue": 1,
                                                                  "green": 2,
                                                                  "brown": 3,
                                                                  "black": 3,
                                                                  # "other": 5
                                                                  },
                                  'high_abv_frequency_transform': {"none": 1,
                                                                   "occasionally": 2,
                                                                   "daily": 3,
                                                                   },
                                  'nevi_count_transform': {"0": 0,
                                                           "1-50": 1,
                                                           "51-100": 2,
                                                           "101-200": 3,
                                                           "200+": 4,
                                                           },
                                  'amount_sun_exposure_transform': {"none": 0,
                                                                    "low": 1,
                                                                    "moderate": 2,
                                                                    "intense": 3,
                                                                    },
                                  'nca_transform': {"no": 0,
                                                    "yes": 1},
                                  'braf_mutation_transform': {'YES': 1,
                                                              'NO': 0,
                                                              'not assessable': np.nan},
                                  'cdkn2a_mutation_transform': {'YES': 1,
                                                                'YES (heterozygous)': 1,
                                                                'VOUS': 1,
                                                                'WILD TYPE': 1,
                                                                'NO': 0,
                                                                'not assessable': np.nan},
                                  'patient_hair_color_transform': {'black': 1,
                                                                   'brown': 2,
                                                                   'blonde': 3,
                                                                   'red': 4
                                                                   },
                                  'cutaneous_biopsy_associated_nevus_transform': {'none': 0,
                                                                                  'common_acquired': 1,
                                                                                  'congenital': 1,
                                                                                  'other': 1,
                                                                                  'dysplastic': 1
                                                                                  }
                                  }
                   }

compute_ajcc = {
    'breslow_col': 'cutaneous_biopsy_breslow',
    'ulceration_col': 'cutaneous_biopsy_ulceration'
}

keep_cols = {'cols': ['patient_gender',
                      'patient_eye_color',
                      'patient_phototype',
                      'cutaneous_biopsy_breslow',
                      'cutaneous_biopsy_mitotic_index',
                      'cutaneous_biopsy_associated_nevus',
                      'cutaneous_biopsy_vascular_invasion',
                      'cutaneous_biopsy_regression',
                      'cutaneous_biopsy_lymphatic_invasion',
                      'cutaneous_biopsy_ulceration',
                      'cutaneous_biopsy_neurotropism',
                      'cutaneous_biopsy_satellitosis',
                      'mc1r',
                      'LAB1300',
                      'LAB1301',
                      'LAB1307',
                      'LAB1309',
                      'LAB1311',
                      'LAB1313',
                      'LAB1314',
                      'LAB1316',
                      'LAB2404',
                      'LAB2405',
                      'LAB2406',
                      'LAB2407',
                      'LAB2419',
                      'LAB2422',
                      'LAB2467',
                      'LAB2469',
                      'LAB2476',
                      'LAB2498',
                      'LAB2544',
                      'LAB2679',
                      'LAB4176',
                      # 'LABGF_filtrat_glomerular',
                      'high_abv_frequency',
                      'height',
                      'weight',
                      'nevi_count',
                      'nca',
                      'nca_count',
                      'blue_nevi_count',
                      'time_smoking',
                      'cigars_per_day',
                      'amount_sun_exposure',
                      'braf_mutation',
                      'cdkn2a_mutation',
                      'decision_dx',
                      'age',
                      'primary_tumour_location_coded_acral',
                      'primary_tumour_location_coded_head and neck',
                      'primary_tumour_location_coded_lower limbs',
                      'primary_tumour_location_coded_upper limbs',
                      'primary_tumour_location_coded_mucosa',
                      'cutaneous_biopsy_predominant_cell_type_fusocellular',
                      'cutaneous_biopsy_predominant_cell_type_pleomorphic',
                      'cutaneous_biopsy_predominant_cell_type_sarcomatoid',
                      'cutaneous_biopsy_predominant_cell_type_small_cell',
                      'cutaneous_biopsy_predominant_cell_type_spindle',
                      'cutaneous_biopsy_histological_subtype_acral_lentiginous',
                      'cutaneous_biopsy_histological_subtype_desmoplastic',
                      'cutaneous_biopsy_histological_subtype_lentiginous_malignant',
                      'cutaneous_biopsy_histological_subtype_mucosal',
                      'cutaneous_biopsy_histological_subtype_nevoid',
                      'cutaneous_biopsy_histological_subtype_nodular',
                      'cutaneous_biopsy_histological_subtype_spitzoid',
                      'patient_hair_color_black',
                      'patient_hair_color_blond',
                      'patient_hair_color_red']
             }






'''
['registry_id',
 'institution_id',
 'patient_gender',
 'birthdate',
 'patient_eye_color',
 'patient_race',
 'patient_phototype',
 'cutaneous_biopsy_breslow',
 'cutaneous_biopsy_mitotic_index',
 'cutaneous_biopsy_associated_nevus',
 'cutaneous_biopsy_vascular_invasion',
 'cb_examined_at',
 'cutaneous_biopsy_regression',
 'cutaneous_biopsy_lymphatic_invasion',
 'cutaneous_biopsy_ulceration',
 'cutaneous_biopsy_neurotropism',
 'cutaneous_biopsy_satellitosis',
 'slnb_total_count',
 'slnb_positive_count',
 'lymph_total_count',
 'lymph_positive_count',
 'cutaneous_met',
 'nodal_met',
 'visceral_met',
 'mc1r',
 'LAB1300',
 'LAB1301',
 'LAB1307',
 'LAB1309',
 'LAB1311',
 'LAB1313',
 'LAB1314',
 'LAB1316',
 'LAB2404',
 'LAB2405',
 'LAB2406',
 'LAB2407',
 'LAB2419',
 'LAB2422',
 'LAB2467',
 'LAB2469',
 'LAB2476',
 'LAB2498',
 'LAB2544',
 'LAB2679',
 'LAB4176',
 'LABGF_filtrat_glomerular',
 'low_abv_frequency',
 'high_abv_frequency',
 'level',
 'height',
 'weight',
 'nevi_count',
 'nca',
 'nevi_colour',
 'nca_count',
 'blue_nevi',
 'blue_nevi_count',
 'time_smoking',
 'cigars_per_day',
 'amount_sun_exposure',
 'braf_mutation',
 'cdkn2a_mutation',
 'decision_dx',
 'temporal_distance_cut_met',
 'temporal_distance_nod_met',
 'temporal_distance_visc_met',
 'staging_doctors',
 'opt_in_slnb',
 'rn',
 'age',
 'primary_tumour_location_coded_acral',
 'primary_tumour_location_coded_head and neck',
 'primary_tumour_location_coded_lower limbs',
 'primary_tumour_location_coded_upper limbs',
 'primary_tumour_location_coded_mucosa',
 'cutaneous_biopsy_predominant_cell_type_fusocellular',
 'cutaneous_biopsy_predominant_cell_type_pleomorphic',
 'cutaneous_biopsy_predominant_cell_type_sarcomatoid',
 'cutaneous_biopsy_predominant_cell_type_small_cell',
 'cutaneous_biopsy_predominant_cell_type_spindle',
 'cutaneous_biopsy_histological_subtype_acral_lentiginous',
 'cutaneous_biopsy_histological_subtype_desmoplastic',
 'cutaneous_biopsy_histological_subtype_lentiginous_malignant',
 'cutaneous_biopsy_histological_subtype_mucosal',
 'cutaneous_biopsy_histological_subtype_nevoid',
 'cutaneous_biopsy_histological_subtype_nodular',
 'cutaneous_biopsy_histological_subtype_spitzoid',
 'patient_hair_color_black',
 'patient_hair_color_blond',
 'patient_hair_color_red',
 'ajcc_predictor']
'''