# agents/chat_agent.py

suggest_t_tests = {
  "name": "suggest_t_tests",
  "description": "Suggest t-tests based on a sample of a CSV dataset. Use this when a user asks what t-tests could be run.",
  "parameters": {
    "type": "object",
    "properties": {
      "csv_sample": {
        "type": "string",
        "description": "A string representing a sample of the dataset in CSV format. Includes headers and at least a few rows of data."
      }
    },
    "required": ["csv_sample"],
    "additionalProperties": False
  }
}

parse_user_request_for_ttest_antigo = {
  "name": "parse_user_request_for_ttest",
  "description": "Given a user question and a dataset summary, return the appropriate group and value columns for a t-test.",
  "parameters": {
    "type": "object",
    "properties": {
      "user_message": {
        "type": "string",
        "description": "The user’s question, e.g., 'Do men and women have different survival times?'"
      },
      "variable_summary": {
        "type": "array",
        "description": "List of dataset variables with types.",
        "items": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "status": {"type": "string"}
          },
          "required": ["name", "status"]
        }
      },
      "csv_sample": {
        "type": "string",
        "description": "A short sample of the CSV dataset to help with context."
      }
    },
    "required": ["user_message", "variable_summary", "csv_sample"],
    "additionalProperties": False
  }
}


parse_user_request_for_ttest = {
  "name": "parse_user_request_for_ttest",
  "description": "Given a user question and a dataset summary, return the appropriate group and value columns for a t-test.",
  "parameters": {
    "type": "object",
    "properties": {
      "user_message": {
        "type": "string",
        "description": "The user's natural language question about the dataset."
      },
      "variable_summary": {
        "type": "array",
        "description": "List of dataset variables with types.",
        "items": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "status": {"type": "string"}
          },
          "required": ["name", "status"]
        }
      },
      "csv_sample": {
        "type": "string",
        "description": "A sample of the dataset in CSV format for additional context."
      }
    },
    "required": ["user_message", "variable_summary", "csv_sample"],
    "additionalProperties": False
  }
}


run_t_test_antigo = {
  "name": "run_t_test",
  "description": "Run a two-sample t-test on the dataset using a categorical grouping variable and a numeric outcome variable.",
  "parameters": {
    "type": "object",
    "properties": {
      "group_col": {
        "type": "string",
        "description": "The name of the categorical column to group by."
      },
      "value_col": {
        "type": "string",
        "description": "The name of the numeric column to compare between groups."
      }
    },
    "required": ["group_col", "value_col"],
    "additionalProperties": False
  }
}


run_t_test = {
  "name": "run_t_test",
  "description": "Run a two-sample t-test on the dataset using a categorical grouping variable and a numeric outcome variable.",
  "parameters": {
    "type": "object",
    "properties": {
      "group_col": {
        "type": "string",
        "description": "The name of the categorical column to group by."
      },
      "value_col": {
        "type": "string",
        "description": "The name of the numeric column to compare between groups."
      },
      "group_filter": {
        "type": "array",
        "items": {
          "type": "string"
        },
        "description": "Optional: List of two group values to restrict comparison to (e.g., ['Systemic', 'Surgery'])"
      }
    },
    "required": ["group_col", "value_col"],
    "additionalProperties": False
  }
}


interpret_result = {
  "name": "interpret_result",
  "description": "Interpret the result of a statistical analysis and return a written summary in thesis/report style.",
  "parameters": {
    "type": "object",
    "properties": {
      "test_type": {
        "type": "string",
        "description": "The type of test or model used, e.g., 't-test', 'correlation'."
      },
      "test_info": {
        "type": "object",
        "description": "The inputs used in the test (e.g., group_col, value_col, or other config)."
      },
      "result": {
        "type": "object",
        "description": "The output of the test (e.g., p-value, t-statistic, group means)."
      }
    },
    "required": ["test_type", "test_info", "result"],
    "additionalProperties": False
  }
}


run_anova = {
    "name": "run_anova",
    "description": "Run a one-way ANOVA to compare a numeric variable across 3 or more groups.",
    "parameters": {
        "type": "object",
        "properties": {
            "group_col": {
                "type": "string",
                "description": "The name of the categorical column to group by."
            },
            "value_col": {
                "type": "string",
                "description": "The name of the numeric column to compare across groups."
            }
        },
        "required": ["group_col", "value_col"],
        "additionalProperties": False
    }
}


suggest_anovas = {
  "name": "suggest_anovas",
  "description": "Suggest one-way ANOVA tests based on a sample of a CSV dataset. Use this when a user asks what ANOVAs can be performed.",
  "parameters": {
    "type": "object",
    "properties": {
      "csv_sample": {
        "type": "string",
        "description": "A string representing a sample of the dataset in CSV format. Includes headers and a few rows of data."
      }
    },
    "required": ["csv_sample"],
    "additionalProperties": False
  }
}


parse_user_request_for_anova = {
  "name": "parse_user_request_for_anova",
  "description": "Given a user question and dataset context, identify whether the user is asking for an ANOVA and extract the relevant columns.",
  "parameters": {
    "type": "object",
    "properties": {
      "user_message": {
        "type": "string",
        "description": "The user's request, e.g., 'Does survival differ across treatment types?'"
      },
      "variable_summary": {
        "type": "array",
        "description": "List of dataset variables and their types.",
        "items": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "status": {"type": "string"}
          },
          "required": ["name", "status"]
        }
      },
      "csv_sample": {
        "type": "string",
        "description": "A short sample of the CSV dataset for added context."
      }
    },
    "required": ["user_message", "variable_summary", "csv_sample"],
    "additionalProperties": False
  }
}


parse_user_request_for_tukey = {
  "name": "parse_user_request_for_tukey",
  "description": "Given a user question and dataset context, identify whether the user is asking for a Tukey HSD post-hoc test and extract the relevant grouping and value columns.",
  "parameters": {
    "type": "object",
    "properties": {
      "user_message": {
        "type": "string",
        "description": "The user's question or analysis request, e.g., 'Run a Tukey test to compare treatment types.'"
      },
      "variable_summary": {
        "type": "array",
        "description": "List of dataset variables and their types.",
        "items": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "status": {"type": "string"}
          },
          "required": ["name", "status"]
        }
      },
      "csv_sample": {
        "type": "string",
        "description": "A short sample of the CSV dataset for context."
      }
    },
    "required": ["user_message", "variable_summary", "csv_sample"],
    "additionalProperties": False
  }
}

run_tukey_posthoc = {
  "name": "run_tukey_posthoc",
  "description": "Run a Tukey HSD post-hoc test comparing a numeric variable across groups of a categorical variable.",
  "parameters": {
    "type": "object",
    "properties": {
      "group_col": {
        "type": "string",
        "description": "The categorical column defining the groups (e.g., treatment_type)."
      },
      "value_col": {
        "type": "string",
        "description": "The numeric outcome variable (e.g., progression_free_survival_months)."
      }
    },
    "required": ["group_col", "value_col"],
    "additionalProperties": False
  }
}


parse_user_request_for_chi_squared = {
    "name": "parse_user_request_for_chi_squared",
    "description": "Given a user question and dataset context, identify whether the user is asking for a Chi-squared test and extract the relevant categorical columns.",
    "parameters": {
        "type": "object",
        "properties": {
            "user_message": {
                "type": "string",
                "description": "The user's question or analysis request, e.g., 'Is there a relationship between gender and smoking history?'"
            },
            "variable_summary": {
                "type": "array",
                "description": "List of dataset variables and their types.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "status": {"type": "string"}
                    },
                    "required": ["name", "status"]
                }
            },
            "csv_sample": {
                "type": "string",
                "description": "A short sample of the CSV dataset for added context."
            }
        },
        "required": ["user_message", "variable_summary", "csv_sample"],
        "additionalProperties": False
    }
}


run_chi_squared_test = {
    "name": "run_chi_squared_test",
    "description": "Run a Chi-squared test of independence between two categorical variables.",
    "parameters": {
        "type": "object",
        "properties": {
            "group_col": {
                "type": "string",
                "description": "The first categorical column (e.g., gender)"
            },
            "value_col": {
                "type": "string",
                "description": "The second categorical column (e.g., smoking_history)"
            }
        },
        "required": ["group_col", "value_col"],  
        "additionalProperties": False
    }
}



suggest_chi_squared_tests = {
  "name": "suggest_chi_squared_tests",
  "description": "Suggest Chi-squared tests for relationships between categorical variables based on the dataset sample.",
  "parameters": {
    "type": "object",
    "properties": {
      "variable_summary": {
        "type": "array",
        "description": "List of dataset variables and their types.",
        "items": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "status": {"type": "string"}
          },
          "required": ["name", "status"]
        }
      },
      "csv_sample": {
        "type": "string",
        "description": "A short sample of the CSV dataset for added context."
      }
    },
    "required": ["variable_summary", "csv_sample"],
    "additionalProperties": False
  }
}


parse_user_request_for_correlation = {
    "name": "parse_user_request_for_correlation",
    "description": "Determine if the user is asking for a correlation analysis and extract relevant columns and method.",
    "parameters": {
        "type": "object",
        "properties": {
            "user_message": {
                "type": "string",
                "description": "The user's natural language request (e.g., 'Is there a correlation between age and survival?')"
            },
            "variable_summary": {
                "type": "array",
                "description": "List of dataset variables and their types.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "status": {"type": "string"}
                    },
                    "required": ["name", "status"]
                }
            },
            "csv_sample": {
                "type": "string",
                "description": "A short sample of the CSV dataset for added context."
            }
        },
        "required": ["user_message", "variable_summary", "csv_sample"],
        "additionalProperties": False
    }
}


suggest_correlation_tests = {
    "name": "suggest_correlation_tests",
    "description": "Suggest correlation tests between numeric variables based on the dataset sample.",
    "parameters": {
        "type": "object",
        "properties": {
            "variable_summary": {
                "type": "array",
                "description": "List of dataset variables and their types.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "status": {"type": "string"}
                    },
                    "required": ["name", "status"]
                }
            },
            "csv_sample": {
                "type": "string",
                "description": "A short sample of the CSV dataset for added context."
            }
        },
        "required": ["variable_summary", "csv_sample"],
        "additionalProperties": False
    }
}


run_correlation_test = {
    "name": "run_correlation_test",
    "description": "Run a correlation test (Pearson or Spearman) between two quantitative variables.",
    "parameters": {
        "type": "object",
        "properties": {
            "col1": {
                "type": "string",
                "description": "The first numeric column (e.g., age)"
            },
            "col2": {
                "type": "string",
                "description": "The second numeric column (e.g., pack_years)"
            },
            "method": {
                "type": "string",
                "description": "Correlation method: 'pearson' (default) or 'spearman'",
                "enum": ["pearson", "spearman"]
            }
        },
        "required": ["col1", "col2"],
        "additionalProperties": False
    }
}


summarize_variable_statistics = {
    "name": "summarize_variable_statistics",
    "description": "Compute summary statistics for each variable in the dataset, including distributions for categorical and numeric variables.",
    "parameters": {
        "type": "object",
        "properties": {
            "variable_summary": {
                "type": "array",
                "description": "A list of variables with their name and status ('categorical' or 'quantitative')",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The name of the variable"
                        },
                        "status": {
                            "type": "string",
                            "description": "Whether the variable is 'categorical' or 'quantitative'",
                            "enum": ["categorical", "quantitative"]
                        }
                    },
                    "required": ["name", "status"]
                }
            }
        },
        "required": ["variable_summary"],
        "additionalProperties": False
    }
}



tools = [
    {"type": "function", "function": suggest_t_tests},
    {"type": "function", "function": parse_user_request_for_ttest},
    {"type": "function", "function": run_t_test},
    {"type": "function", "function": interpret_result},
    {"type": "function", "function": suggest_anovas},
    {"type": "function", "function": parse_user_request_for_anova},
    {"type": "function", "function": run_anova},
    {"type": "function", "function": parse_user_request_for_tukey},
    {"type": "function", "function": run_tukey_posthoc},
    {"type": "function", "function": suggest_chi_squared_tests},
    {"type": "function", "function": parse_user_request_for_chi_squared},
    {"type": "function", "function": run_chi_squared_test},
    {"type": "function", "function": suggest_correlation_tests},
    {"type": "function", "function": parse_user_request_for_correlation},
    {"type": "function", "function": run_correlation_test},
    {"type": "function", "function": summarize_variable_statistics}
]



