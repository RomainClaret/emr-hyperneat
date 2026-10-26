"""HMR's values that EMR's tests compare against, recorded from research's frozen HMR engine.

The public release's tests do not run HMR. A test that compares EMR against it asks
hmr_reference.hmr_value for HMR's side, which returns the value recorded here for the test's case
and the sha256 of its inputs. Research's scripts/release/record_hmr_reference.py wrote this file: it
ran research's tests that compare EMR against HMR live, on the CPU, and recorded only after every
one of them had passed. Values are exact (arrays by dtype, shape and zlib-compressed bytes in
base64, floats by float.hex) and kept per CPU architecture, with the research commit and the
versions they were recorded with, and "xla_flags", the XLA_FLAGS conditions they were recorded
under (unset, and the nightly's --xla_cpu_multi_thread_eigen=false: on x86-64 some inputs EMR
computes depend on it); "agree" says whether the two architectures recorded the same values for
the same inputs (None until both are recorded). Do not edit by hand: record again.
"""

RECORDED = {
    'additive_settling.fulls': {
        'platforms': {
            'arm64': {
                'research': '7440581b174be13887db6c391e45efc9c3ee2668',
                'recorded': '2026-10-22',
                'jax': '0.6.1',
                'jaxlib': '0.6.1',
                'numpy': '1.26.4',
                'xla_flags': ['', '--xla_cpu_multi_thread_eigen=false'],
                'values': {
                    '00a0120a55f89ae56a7099e32b83652751519f839e54cbf2dcb5b78312fc9f10': {
                        'tests': [
                            (
                                'test_additive_settling.py::TestMultiTaskAgainstLiveFull::'
                                'test_equal_on_the_same_arrays[nt1]'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 6,
                                        'substrate': {
                                            'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'max_depth': 2,
                                            'variance_threshold': 0.03,
                                            'recurrence': {'preset': 'hidden_only'},
                                            'neuromodulation': {
                                                'mode': 'true_neuromodulation_4nt_option_a_tanh',
                                                'modulation_strength': 5.0,
                                            },
                                        },
                                    },
                                },
                            },
                            5,
                            [
                                '<f4[6, 2, 12]',
                                '<f4[6, 12, 1]',
                                '<f4[6, 12, 4]',
                                '<f4[6, 12]',
                                '<f4[4]',
                                '<f4[4, 2]',
                                '<f4[4, 1]',
                                '<i4[6, 30]',
                                '<i4[6, 30]',
                                '<f4[6, 30]',
                                '|b1[6, 30]',
                                'mse',
                                'full',
                                None,
                            ],
                        ],
                        'value': {
                            'array': '<f4',
                            'shape': [6],
                            'zlib64': 'eNrrWWFo16CyzIaBQcGs4YKVjbp3nf2W7hY7AGbpCHY=',
                        },
                    },
                    '756bcc09f7ae10bb6533d71a73776ea9b0f5f0b3289189a5189c4989cb47ab45': {
                        'tests': [
                            (
                                'test_additive_settling.py::TestMultiTaskAgainstLiveFull::'
                                'test_equal_on_the_same_arrays[nt0]'
                            ),
                            (
                                'test_additive_settling.py::TestMultiTaskAgainstLiveFull::'
                                'test_negative_control_emrs_rule'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 6,
                                        'substrate': {
                                            'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'max_depth': 2,
                                            'variance_threshold': 0.03,
                                            'recurrence': {'preset': 'hidden_only'},
                                            'neuromodulation': {
                                                'mode': 'true_neuromodulation_4nt_option_a_tanh',
                                                'modulation_strength': 5.0,
                                            },
                                        },
                                    },
                                },
                            },
                            5,
                            [
                                '<f4[6, 2, 12]',
                                '<f4[6, 12, 1]',
                                '<f4[6, 12, 4]',
                                '<f4[6, 12]',
                                '<f4[4]',
                                '<f4[4, 2]',
                                '<f4[4, 1]',
                                '<i4[6, 30]',
                                '<i4[6, 30]',
                                '<f4[6, 30]',
                                '|b1[6, 30]',
                                'mse',
                                'full',
                                None,
                            ],
                        ],
                        'value': {
                            'array': '<f4',
                            'shape': [6],
                            'zlib64': 'eNpb8XuZ3Su5QnsGnXtWHTxJdv/+19t78LXbAgCOpQqN',
                        },
                    },
                },
            },
            'x86_64': {
                'research': '7440581b1',
                'recorded': '2026-10-23',
                'jax': '0.6.1',
                'jaxlib': '0.6.1',
                'numpy': '1.26.4',
                'xla_flags': ['', '--xla_cpu_multi_thread_eigen=false'],
                'values': {
                    '00a0120a55f89ae56a7099e32b83652751519f839e54cbf2dcb5b78312fc9f10': {
                        'tests': [
                            (
                                'test_additive_settling.py::TestMultiTaskAgainstLiveFull::'
                                'test_equal_on_the_same_arrays[nt1]'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 6,
                                        'substrate': {
                                            'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'max_depth': 2,
                                            'variance_threshold': 0.03,
                                            'recurrence': {'preset': 'hidden_only'},
                                            'neuromodulation': {
                                                'mode': 'true_neuromodulation_4nt_option_a_tanh',
                                                'modulation_strength': 5.0,
                                            },
                                        },
                                    },
                                },
                            },
                            5,
                            [
                                '<f4[6, 2, 12]',
                                '<f4[6, 12, 1]',
                                '<f4[6, 12, 4]',
                                '<f4[6, 12]',
                                '<f4[4]',
                                '<f4[4, 2]',
                                '<f4[4, 1]',
                                '<i4[6, 30]',
                                '<i4[6, 30]',
                                '<f4[6, 30]',
                                '|b1[6, 30]',
                                'mse',
                                'full',
                                None,
                            ],
                        ],
                        'value': {
                            'array': '<f4',
                            'shape': [6],
                            'zlib64': 'eNprWWFo16CyzIaBQcGs4YKVjbp3nf2W7hY7AGYpCG4=',
                        },
                    },
                    '756bcc09f7ae10bb6533d71a73776ea9b0f5f0b3289189a5189c4989cb47ab45': {
                        'tests': [
                            (
                                'test_additive_settling.py::TestMultiTaskAgainstLiveFull::'
                                'test_equal_on_the_same_arrays[nt0]'
                            ),
                            (
                                'test_additive_settling.py::TestMultiTaskAgainstLiveFull::'
                                'test_negative_control_emrs_rule'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 6,
                                        'substrate': {
                                            'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'max_depth': 2,
                                            'variance_threshold': 0.03,
                                            'recurrence': {'preset': 'hidden_only'},
                                            'neuromodulation': {
                                                'mode': 'true_neuromodulation_4nt_option_a_tanh',
                                                'modulation_strength': 5.0,
                                            },
                                        },
                                    },
                                },
                            },
                            5,
                            [
                                '<f4[6, 2, 12]',
                                '<f4[6, 12, 1]',
                                '<f4[6, 12, 4]',
                                '<f4[6, 12]',
                                '<f4[4]',
                                '<f4[4, 2]',
                                '<f4[4, 1]',
                                '<i4[6, 30]',
                                '<i4[6, 30]',
                                '<f4[6, 30]',
                                '|b1[6, 30]',
                                'mse',
                                'full',
                                None,
                            ],
                        ],
                        'value': {
                            'array': '<f4',
                            'shape': [6],
                            'zlib64': 'eNpb9XuZ3Su5QnsGnXtWHTxJdv/+19t78LXbAgCO1QqP',
                        },
                    },
                },
            },
        },
        'agree': False,
    },
    'anti_node_rule.antis_arrays': {
        'platforms': {
            'arm64': {
                'research': '7440581b174be13887db6c391e45efc9c3ee2668',
                'recorded': '2026-10-22',
                'jax': '0.6.1',
                'jaxlib': '0.6.1',
                'numpy': '1.26.4',
                'xla_flags': ['', '--xla_cpu_multi_thread_eigen=false'],
                'values': {
                    'adc7cbea87493ea1399fa2e2a0a398dce751c313a4959615483b5dc63bf15d31': {
                        'tests': [
                            (
                                'test_anti_node_rule.py::TestAgainstLiveAnti::'
                                'test_equal_on_antis_arrays'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 10,
                                        'substrate': {
                                            'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'max_depth': 3,
                                            'variance_threshold': 0.03,
                                            'neuromodulation': {
                                                'mode': 'true_neuromodulation_4nt_option_a_tanh',
                                                'modulation_strength': 5.0,
                                            },
                                        },
                                        'neat_species': {
                                            'compatibility_threshold': 2.5,
                                            'max_stagnation': 40,
                                        },
                                    },
                                },
                            },
                            {
                                'problem': '_TwoInputXOR',
                                'get_data': [
                                    [[0.0, 0.0], [0.0]],
                                    [[0.0, 1.0], [1.0]],
                                    [[1.0, 0.0], [1.0]],
                                    [[1.0, 1.0], [0.0]],
                                ],
                                'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                                'input_shape': [2],
                                'inputs': [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
                                'jitable': True,
                                'output_coords': [[0.0, 1.0]],
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                            },
                            42,
                            'xor',
                            '<f4[4, 2]',
                            '<f4[4, 1]',
                        ],
                        'value': {
                            'tuple': [
                                {
                                    'tuple': [
                                        {
                                            'array': '<f4',
                                            'shape': [10, 2, 340],
                                            'zlib64': (
                                                'eNrt23tQVNcdB3A1mPhoMZoERSPRKBJUJEZLfEz2/kLwiWiCKI'
                                                'IJKFrDVKyv0FQbLYhYggpGJPjYAIqw8lZhebP3ew8aRo0gHbI+'
                                                'qFZAjA6WiDwsaKY9caZJp4wzwD+mw+8z8xuWe/Z7z93LzuF3d+'
                                                '7O3ZlH+xYtJ0d9KC0c6oCcFhOtjMmlgreP0d2RbpTyUT7Z+Rko'
                                                '4qA/lW6ZoltRmEoJzkHUcmSJYmf/DubtC6d29yLlzuhXsXHtQf'
                                                'RijLEeauZFE/nUxVPIhRQyWDoq5cs1qlxvIn+rHFp05SuyHKNS'
                                                'tHUepZ5OoLerPcjtQj7NPXqCPvFbTx+6GXXXStNIPzmE/McEK1'
                                                'X6qbyeMsZ6rLrx7srLramwX7peveWTjY9y3WnwkEOq4yv7YeFl'
                                                'RODv7alydy9k9dUDw/PgVt+kBOocEfDPBDy/oABzCsMVY7onYh'
                                                'ek4OqAIl5PGWM9VkaFoLKJGgUUCTo4AfSnOkEueYI2jRA0qk2l'
                                                'hpuCgk8Iyr2lUex0lZyrBJkPCPKK18g8z0QOlYLmfSbo/nyNVo'
                                                '4sJj6jjLGealVDLH17z4ghFwvRcFvAajjIq9dxihqaoUaUZUEb'
                                                'cIheL7bGKlMuEr/TUGLljjOJBbByEHj3loDFhSKcXyGguyuwrl'
                                                'X81J/auxWQ13tJ5DrElUqmh6v/O++lyKNUvsuRPF6pUC87HPgp'
                                                '5z07l6KH6SljymA60ti/Q79rq0XTwhM3Ffz9DVisju8w3hR3WH'
                                                'kwaNZT+2TDGh80rU9+Zn30hr37qHnHB1ThlaH0eWGD2tX8hc0u'
                                                '9PG2cGWis16taZvf6dehrg2j0nQnGl3pqtzLVrs8r8OUN6jtcG'
                                                '8lbGqN6pe+qtPzDnPYThn9rMlOeOv+XN67y+c94zf9n/yP/q5+'
                                                'MBKjtnQ671bnT9FhjcrYd1TTi6tHd3lei5ByxTR1vLrM8U3c8N'
                                                '/V6XyL+X2avjVNeVS7Wb0+R9fleV+KClNqx8Wqs1JcceViBF/v'
                                                '/Z+JeyioXPagK8oExQw6RRXy90RZ9S2CmmsFRcvHJB/bXxcUWK'
                                                'vRimZBA2VPatypUegXWVRWKsizr0ZbxmRQq+tw3X/2e17mUmTN'
                                                'lWXZKjr0rQly+1uySuS+JzX+PH5GbtPLspa1saVjbr/c3i73N0'
                                                'iO3WnoOG6WY3HN4ql98gg5VlUvnlkfHS6P303WTXmcr7Z2/Tic'
                                                'ZbZM5h7L1x/S3Pn8dplzklUgs5dbuj7veJk9LrOnZfZxU+fzG2'
                                                'TORtYemd3VjXkHy+xmmfWW2ZVdmNdX5vrI8pJZ+27M2yhzM2W1'
                                                'yHOc9qDzedf/+tsWNXd93osy90ge7/Yf36eNz+59yrpnRtJphP'
                                                '4rDhFxC/DD2XHqX2cWwHdmDlKXn0RuvgH+AzKw1v04nEd8iT1x'
                                                'wei7Zx+8gzbDtWUqXrM9rwZY2sB2fJTad2G8zu61GwqfUcZYT3'
                                                'UuXUNfZ8DiTCHWGbLw0iyB6mYNB4M0XLsMWC3WcDYaiN2hYuSD'
                                                'YrlBxV2PYiw8UwD/53IxSOTDHGaEx5JMeE5L4usTxliP5fjpMe'
                                                'w9eRpXLQ1Ibcvm9ZAxxrpp3V2BrfcF1tTLn42C11PGGOumx9NG'
                                                '4/7Rgch019OcCfFkMyEU4l4wllhuQ8yDQMQX55g2nxpp+vG5B+'
                                                'YG6z6z30Sfrt9C308Oog93hlBgdToVN2fQpbZMWtd4kviMMsZ6'
                                                'Ks2cTAM90+hYDcjfSqOVW/ZT6PcHSJ0dQ95Bh8hvhpGy7xnpRn'
                                                'QOLXXKpVl9VGqYqlL/36qUv1+lYZc0On9DI+2eRi+0a7yeMsZ6'
                                                'rDGLj8HjcyOeK7HH8b9FwXZyIh5WZ8KYnYMPBhfC5r3duDQpDo'
                                                'sGpOE+ZWPK6zNhMz8Ik9pj4PM7A8Zum6fqVwyFbeUaBPpE8ucF'
                                                'jLEea+mRneivJGP686dMNxI8sLX9c/gti0VrUCoW3s1C+TUn+O'
                                                '7dhqC2L2FzNgnf3HlLjYwZgtQFq2CYGIEdwZef3CPVe9y3al6D'
                                                'C6+njLEea8YXnmpJaJRSXGOkc2MLqXlrALJnOONMyiD0+0OqGp'
                                                'lYobQeHkXKvvfpzUl/JEPAMYqsTqarsZn0db9s+tWIInqYVEyP'
                                                'rphInFU7XO+bm/yQXOWGSclOsFs9ssN6W3p7t+qWbv3k+zoNfy'
                                                'l/8n2Aqj5m3dOONznTgT5JnMWfKzDGfnFcwgRadQKjTufj15uy'
                                                'cO2cgKFIICVVoCxaYI2twK07GhoPa9jqpMHsqOLdoSb4phVhr7'
                                                'kAE3WncOSHVNTYJ8Jqkb7DevmPrwW8IVCXJ7DhZMf7ByhSwDFY'
                                                'YNpGgeG+Auc2CHzl8/T7DIRZw8VMjftgxtgvTqmiUeD1aDpqEB'
                                                'TxciH3fYwx1k15i4poWe1e5XaJRnXZCbyeMsZYN5nXJMEWs9Hk'
                                                'cgoXAnbxdTRjjHXT48UaTNuL8fF9Dd67VV5PGWOsm/4NllTK2Q'
                                                '=='
                                            ),
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [10, 340, 1],
                                            'zlib64': (
                                                'eNrt2H9MlVUYB3AUxQYhzDUH0VzaaJEk1yBoQfd9colMh6wRbJ'
                                                'opk3VNu44wx8IABRY6uiUkpnTRoLuLGr/uRQTuD+75vs9MqUQo'
                                                'YTBgWKC0aBoNBw2ldecf/iFzA/6pxvn8e3bOc97znvc5z3ltMU'
                                                'UonnwHdclautybR1OR3+B5iwkJ+4yIX16MrB2pII8YdN3yxZ6u'
                                                '8yLm+EIlbcCqpHv7Ut0r0ZTzWz71/VxEZzYY6c3PTOQhSZI0T6'
                                                '29dkOUXW6mc1f8yZ4t6MWNxzFqWKH0tlVQwuEWCrm5G61hz9CC'
                                                '1jq61CqoKNQLX5Rk05bOJuoPVO/nTyMZqTTTSRHXVZlPJUmatz'
                                                '7pZFT3mHFHJ7Cu4wDVjDEuORk+1QK/ZBdDDDLGflCxvsuKAp9G'
                                                '7W0DI+4vJ2xPJ6IQpeTXAzToTIh9YxWZVzse5NOjQwx9FePDbh'
                                                'V3rS48HDfse4ZlE8OsBzK0tgftJ0cZRf2Mx4oZX36gTutX+Kt7'
                                                'XDsjxJ/xQqGY1h58jXFoL+NRzxtpZkwKFf/WenslMESdioRMIP'
                                                'hJ16znkTKpYsBbRWiUwPYKx4z7LzrGWBDBWPWRilPLMOu4eSmM'
                                                'iz+qsDqAl8pmPu8yB2Moj1H5BMNfO/t133+KsSmOUfG1CvuOmc'
                                                '+7170P7lkZYe8z1o3OPm6P6t5DRxmaUMbv7828f7L7ewm6ymiv'
                                                'ZGRu5VnHLelmRNoY0TmMEh+GzFD/L150hJrTB7VZVhO2J9mhmb'
                                                'BQUmk5nbiQSxlrAmi84TUKWvKW4unfJh533/0nohVM7f8US4Yr'
                                                'Ua40oLTJgvptNuwabEHa65DvX5KkeWulRqDHfZ7eThG40SHPQ0'
                                                'mSpLk6cFpQVSjT6XcdpGhU2uAjyOinkqc3U38+05/LWuhuEMh2'
                                                'TKX1G5n0gXaqr3FR4iBo0R2VFp+8QFc/dtJ4tyCxT/4/lSRp/i'
                                                'rYFqD0+Q0rBYbNsNT6I+LeWcUzbkwZCXmKUsujKXgyVmicga60'
                                                'FePa1IsGpSR/Ney7vHF94IpYmHhIjJXlIndSj78NiTjbGSnrW0'
                                                'mS5q32jqb7NWXWWju1Je9UdGktFL7nHLXrVlJN5quIW+oiU1I1'
                                                'daovU+3BZxFudNGtkVqKjd9Mb3/ri6gAQX+EW8nhuZO29A2Lh8'
                                                'dvtKeQQVeoHC4cEUuP5EzLt6aJ3bR3zRklPaJd3NyagamDgkY/'
                                                'dzyyzl1+XtBXHk5ZB0uS9J+j7Wf6SWWKYjvFV9XLPCVJkjRH+u'
                                                'ZGfLcYiHmuEeYhIe/rkiRJc/QPKwHoPA=='
                                            ),
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [10, 340, 4],
                                            'zlib64': (
                                                'eNoU131czff7B/CDtBCiKElCulPpvlOdz3UdhsXcxDLNt1mzbM'
                                                '2vkblrFg4yIcRCCCEkIUnC+VzXQUiS3GRROBoWwxptmrXtd/XH'
                                                '85/zeJwePc77el/v18uQ5o5L09zZwW0dOLqto8beC0FQ0NIlpF'
                                                '26BAomBOGRCUHsff8UCBqx7RUI0pzYGyHOzjP64FyjD2f9sAe2'
                                                '/7CHei7siYKjXPvAWNc+pIVCEHR7fXeqXd8dDLbXjULnNr6BPM'
                                                'Y3wNE5gSg4+WExLHpYTBnL++PG5f05LuUriE/5iootn4Igl5up'
                                                'Rs3NVF1tkQZuF2lojLmExppLwGJ7KHbYHsreUy6Ax5QLNMnWEw'
                                                'UnuG+Cme6b6Okf76FgQ+hZRaj6nmkgKD50BiWEzoCYveEoOLVP'
                                                'FSzvU0WN1n4ouHjsISgZe4j0RnsUbPDVgqDkhRdhwcKLlGEgVS'
                                                'gWo/1QsIVDPrR3yKeoM24o2HrdGhCUOdIBfxzpwE3TI6B5egTN'
                                                'MrZHwZrA5UpLwHI1Lf8y/JB/maIiilWhlI+fCYI0P65sA/rq3o'
                                                'pQl6+8T6kr74M5y5UE1B87QAKM/wTg2X8CuDnqBAgqqvFCwVnP'
                                                'tsL2Z1tpfLIzCjZZfw5G68/JprELCnbR/aIINWnLIxCk/+i+Ue'
                                                'hSbm0BQTZbR5HV1lHgYniiOBieqPEWKgkwuRWpQtE5lZOAhrHB'
                                                'KNg85iw8GHOWnKf4ouDKtfugYu0+8rg8CD0vD+ICl2UgKPWOLa'
                                                '68Y8tRbgNgrNsASl7cAoJMnn/ohNGl+CQ4F58k/ZQObaA6QmlD'
                                                'hbCDCmAHmPK0RkOeVmfs3EgCUkeGomAb3/PQ0/c8maP9UfCL3C'
                                                'PwPPcIXYryQMHlRzdA5dENFJTviMH5juxSMRpcK0ZT9vuWuON9'
                                                'S25am6cINSPjBgiy+WKlKpTWu8tAUMbdOW3AkJqqCLUn3yA7vg'
                                                'HJzVpc2Kxlk3M5kHM5jW0IxA8bZJ6/KIEFX5RQoZ83Hvfz5oLp'
                                                '2ZA/PZuiVrrgxJUubL74DQjK+LMbbvizG9c2W4CgxtjnIEjTeP'
                                                'KsiEhO2wsL0vaSX7YPeWT7QF56NyhI70bhRYWkKyqEvPhwPBgf'
                                                'zq26a/BOd41au4fg391DOPq1CkLu8FAMXTqU02cdBEFeaYNRcE'
                                                '7JDyDocXJvFOxw2A/sDvuRi4MGBzhoWHPtQ6Wp8kPVHGUCQTGx'
                                                'd9S42DtKU/EUEFS0dz0V7l0Pcbsj8LPdEWy3/wb03H+DHF+Eom'
                                                'B9chkISpsYgKsmBrC5/XF40P44RVZ7ouDIjEwQ5P3SCQVnbJgC'
                                                'mRumUFN2JxSck35NEarGug4E5fQYqgolLXA9CPLeEUMeO2LAUK'
                                                'BDwdWza6Bqdg1dWh6Ggl/YV0CTfQU9Px2Egs1jT8ODsafJItkH'
                                                'BY/dmwOCus0fiIJLG78DQWV1PVBw+a5eULarF1kZX4MgQ8j3Om'
                                                'FMrD0CgjLNdpRhtoPmJH/8I8mfDRZHYYnFUUr4xgcFNy3JgRdL'
                                                'cijhew8UHBSWAYJa4gfiX/EDOWnFdyAo9bkjrnjuyKYJH0LZhA'
                                                '+p8EIPPH6hB+tte4GgvL2WeHCvJRfuKlBKdxWofgteQ8AC+X/+'
                                                'mK3T/DnbmJN9E/Zm36SmaQZVKMnfHQFBmUW9KKuoF9QHrWhDBq'
                                                'fZbcDKwrMNFR3MpeKDuVB4dI0iVPO8ano8rxpyAixUoUR9/ZAE'
                                                'ZAy0JAG184tJgM75S9I6fwmRe1bRiD2rYNH5ABRcsLIIBI3pOh'
                                                'QFO/y2HwQZv/dC9Xsvtjm/BQTF17mi4Pr1qSBoTFM/FJx06VOY'
                                                'd+lTCprfC4Pn92KPw14gqLxfF7zSrwsbpj5QkqY+UK20/4Agl7'
                                                'HeilCTox+CoCRNR9Wg6agkBBtBUK3/M9Xs/0ypvfMjCMq7P7YN'
                                                '6BPGgS5hHBVkymeZP4Jfyk+KUKf1NFFcTxNoFqoR4mz7kc/JYu'
                                                'Rz8BtpUoUyvPdFGtH7IsQvGE4JC4ZD1eUsqr6cBSPeBaJgj5IS'
                                                'ELRgnR8KLvsvHy79l092k73RdrI3a+uyIaQumzwGuqP7QHfOq1'
                                                'sLuXVrKcHogoKz+s2G7H6z6d0jBxTsYI0gqDqyOwqOvGoJUVct'
                                                'KXtvBxQc9WytIlSLMb+CIM2RyWc1RydHWPlWQGffCsqblacWzs'
                                                'pTnC7vA0FNz73agEdpUhsa//VyGvv1coha3aMNjZl6jMZOPQZx'
                                                '3wQqSd8EqhXb7lHltntgiP1SFUrm1BoSEHmpHwnIqzpEAqIcg3'
                                                'GCYzAH7TgDATvOUHSTP37U5M/FJcdAUExnX/yksy+b1+yFhjV7'
                                                'qVTniad1nhx/dBNMP7qJKicNwquTBnFc3FKIjVtK47s5oWCTMg'
                                                'nKlEmk8bBFwTle/SDbqx8ZJlmh4IxQVoQatexPEKQxXdW5mK4a'
                                                'Gz//CZ5+/hMZ1EmqUJJai0BQVFIniknqBH4JqyEgYTXN+346JX'
                                                '0/HVp+CGlDdl/sagM5PQ4qQvVNriDv5Aown9psNJ3arMv1fUJ5'
                                                'vk8g7dsm1fBtk/Ju5xlq3XkGklxDcLZrCBvDCAQN1Aai4PQ+xb'
                                                'C+TzGN2DUU3981lIN+OgghPx2kWN8h+D/fIZzQdRvEd91G+lOD'
                                                'cdipwRw7bhVMHbeKpv/rjILjRseDoFLujae4N0dNCABBeces8e'
                                                'Axa256r0nRWDWpTUs1+NtSDevnRCt+c6LV2hs/gyDzy0Kj6WWh'
                                                'rjLpHAjK8L6hCsWqYhtYVmyjmI/0baBl9dQ2NHbEujZQnfxGEW'
                                                'rem9MkQO++V+fivtc4fcNjit/wGDL+26cKpdn5KgmIcQzFKY6h'
                                                'nLLlHCRvOUfNXwah4AKfUjjqU0pxtv74ma0/e18qAN9LBWT3zh'
                                                't7vfNmzX+74N9/d1HsS3f89KU7pySuB0GjSgag4KSw+ZAYNp9G'
                                                '2DniSDtHzsNRIGgE2eD7ZMNJ3t1BkNeXHVFwda/dilArDb+BIE'
                                                '3piwjNqRdnbQZcB0Ha9htVv/YbldgFh0DQ41XOZF7lDE0zF7ch'
                                                'O+9kcvBOhrRh/dtQ3oBDbcBlbaLitzZRXVBym5JLbkNUuI8aF+'
                                                '6jhOjqKEhXBxn/hOLGf0K59mQZ3DlZRqVLgvH0kmD2azkLAS1n'
                                                'yTIxAAWnfnocBJXt9UXB8atzYfrqXDK0eOLSFk82224GQfVhri'
                                                'jYmL4cBDV59sPfPPuxucMn0NjhE8pZbod7lttxjP1giLYfTC5v'
                                                'O6Hgpu9uKEKNn/sOBJlduivVLt3VTOt6EKR5f7AqFO2MUgiZUU'
                                                'rlq96p1aveKYmpG0BQuXlyG6j95f025Ngni5z6ZEHSRxcUodrM'
                                                'KiMBmuNrz4qIqIbnFN3wHLZc16Jg6+zL0C37MmmXhWDYshBuXc'
                                                'zw72Km+PxAnJEfyGV7T4IgmyF+2H2IH+cVHQJBt+uGYE3dEDaP'
                                                '3gENo3fQvXQ3FFz+bjVcereaHli4oGDz/plQu38mubRzQMGlV8'
                                                'Kg+EoYmUO64cOQbhx3TAOCyqe0x8tT2rP+1feK36vvVevgRhBk'
                                                '0AQYhS574yUQlBV7Ws2JPa1k5O4GQbk1gZRTEwgxqV+1oarff6'
                                                'Dq338Azb73wGLfe5RUcaINaBI6KzYJnVW/8gfkXf4AijaGoWCL'
                                                'cRXQflwF2SWFom1SKOdcOA97L5ynH6qDUMh+PA3aHacpcbM/fr'
                                                'NZ9qPhKJQajlLMdh+cst2Hy+32wCW7PWQq8sBzRR7s8DIDer/M'
                                                'IJtDA7H7oYGc4Pc9CLKCvtgJ+nLamHEg6IBnTxScNt0e0qfbk7'
                                                'fzeyjY8OK4kvziuGqz6w0I0tRu0Jl/2mBM+OsWfPnXLco5OFcV'
                                                'SmOXYyBIH9qTIkN7QnNLahu5G//XBpLrvNtQ7qO9lPNoL1QbNy'
                                                'lCbV9WRRZlVeA6MhwHjQznhMpKmFlZSZVjtXhtrJaTr12ERdfk'
                                                'zTQHo+DGD1V4+qFKHo8D0PNxAAf9VQSCqkKGomDjIukyiw6Q+a'
                                                'gXCtbM3AoWM7dSjONg2S+DOW/KSsifspKyFWcUXLj3MxBUeasX'
                                                'Xr3Vi0vvekPxXW8yZXTBcxldOGfGz4pQjWP/BUEaNVSxUUPV6m'
                                                'wzCNIfaDIKXc4JFXadkJw8rUHVT2tQXqRtBkG140e3gULXiVDg'
                                                'OpGaKza2gepZDxWhbppClDmFwOrXcBRc2u86nOl3nW46hKHgvN'
                                                '3lICimLgSn1IVw+mkTrDltovg+QfhFnyDOGnAKBFWd8EPBpmWH'
                                                'QVBUqjdOTPXm2oU74d7CnZK73XFKrDsnbU0HQU0dBuBvHQZwcn'
                                                '0SCMoN64P7w/pwjuMwyHMcRnknuuOhE905eWgnmDe0E9EgCxRc'
                                                'HbdRqY3bqEYteQHjl7wgTcuG0xpNRljZ1qsgKOniHlUoVl/uB0'
                                                'HFX7pT4ZfukDN5XhtK2mVoA0mze8G82b3oztYjVLtVsnBqBH6T'
                                                'GsF+I26AoPiaMJxRE8bNylV4rVyl2Auh+L8LoVzceAGKGi+QTW'
                                                'gwdg8NZuv/zoAgsg1AwXlDC+HA0EJqDfbFf4J9+bHTPmhw2kcp'
                                                '33ri4m89ufLTH0FQc+YgfJM5iDOWGWD9MgMt+cwJBXv4RIOggL'
                                                'W2KLj8m/4gqLrCCm9UWHFTYpliTixTrfe+BUH64T/rNMN/Njqc'
                                                'qgXHU7VkqhypCiVlVTEIannXkTR/d4TqwWuhavBaim+cRgmN06'
                                                'AlOrwNTd+RTXE7smFETx2O7Knj28NvgaDSFeEouPXwNdAUXKNL'
                                                '27QoOPP+Jdhy/xJpMAQFp24hWL6FqDYmEAVbrymGzmuKKfPSUB'
                                                'Qc9FkeCLKMG4KCnY5vA8fj2yiheTB+3TyYx/ZIA0E/jO6PgvMc'
                                                'v5Tz/5KK+9rjib72rPk2CFrnBFFm/64o2PCuWUl716zWW7bDOs'
                                                't2bC76TDEVfaaW9XwCguKebDUanmzVGX45D4LyvqpQC7+qUDzm'
                                                '7wBBjy/ryHxZB0kR09rQve/WUu13a2HeZh0KnudUA4IaXSPwF1'
                                                'eZh5fXwffldfrzszAUrM2+AiHZV6ghIBQF5704B7kvzpHvyiAU'
                                                'rN9SCoKytf4o2DTiCJwfcYRa3H1QcO7W3bB3625qGOyBgnMGbY'
                                                'C9g+Td+nsAXv57AFe/XAC3Xy4gw9eOuPRrR84aHgmCdG49UHHr'
                                                'wUmpPUBQ+8cdUbDN/FzFan6uevtIEwjSbHXTCaN2RTWErKimqJ'
                                                'lrVP3MNYp1t8MgKNeqL+VZ9QW7JYY2ZNoyrw1UdlPwajeFS2/d'
                                                'gZJbd6jpdAT+djqCcy1uQo7FTfKwD0dP+3CO8agEQcPttCi4vu'
                                                'NFEHRgezAKjv/ECNM/MVLi+gAUHHnrOAhyu+aLgrPt94OgUm8v'
                                                'POXtxY20GZ7TZnJZ5IoDFrmy3ScroPcnK+hecj8U7NI4FQS5/G'
                                                'yHgg3j3SF1vDtFJXbGCYmduXDLHUWoduv+BkFxJxwVoUZOvQ+C'
                                                'mpIdVM13Dkp142kQZJP/RhVKbMRGENQ6QbJ51CSwWqqgYJv/qw'
                                                'VB59/XoeDoP26BoDRTOK4yhbNhZRUISn+sxXWPtRz99DLEPL1M'
                                                'ljtCUPCIWobIWibvikD0qQjksTYlMMamhEIm+aFgl9B8cAqVPm'
                                                'Utfcram5Oe7YBvn+0gfbkbDit346bTa+DV6TX0eJwL/jzOheMS'
                                                'E0HQpqkOKDjLHAGCbu7rhoLjnDqAoKi69jixrj1rDCsUK8MKNW'
                                                'vWMxCk2aYxCl2ZphwEJR0sUoWSF7kHBDmN9yOX8X6QdMIaZ5+w'
                                                '5sKAJsUU0KT6bfBDwdkdD8OujocpGt1QcF671VDQbjWFBIShYO'
                                                'Nf5SAoBR6BoJxBj42mQY9102J7omCb1Q7gsNqBWvLd8G2+fP9/'
                                                'a0BQ3uIAPLg4gM3bjoOgx4mW+HOiJedE5StZUflq9o7+uGNHfy'
                                                '63T4BK+wR6PMMXn8zw5bHqPhiv7iPtW8lnb0N4bN9zIEh/vTcO'
                                                'u96bsw4HgCCnUZ4oWHNtE7RWbiLzf4H48L9ATr1SAsuvlFBSSD'
                                                'jODgnnF1sroXFrJTneH4iC85JToCA5heaN9sO5o/14Xp98WNAn'
                                                'n8Z316LghOYyiG8uo9xNOhQ8tqd0yZ41ZHxiBEF666dqlPVTpf'
                                                'rkMxBkKHl+VlPyPGJavSUKLnQ4ppgcjqnr/7JDwYabMs833cnm'
                                                'QX8UnFX8NQj6ONADBY+YsAGGT9hAi9J9UXAu5MJeyKXquEC8ER'
                                                'fIyfuLYcH+YjLPqAVB+oZIVSgtv/0H7377jzQDRitml9Hqne3d'
                                                'UbDLHivw2GNFj4198YmxL5ePi4KycVEyj4NlHgezk9dKcPFaSc'
                                                '2DvfGPwd7sgtkgaNo6fxSsm3EU9DOOUsWhYBTssN4IjuuNlGxq'
                                                'ggUm2T+Znjph3JXbCQW7NFxTHBquqU8H2KNg85YgeLwliIpWDk'
                                                'DB5Vaye6zmUcICT/x6gSdnzfwRBEUtH4qCW5MPwr/JB+l23yAU'
                                                'bOV1CgQ17QjF33eEctOnF+D1pxeosnMHFBxlWKkI1aZdT+zRTu'
                                                'ZvYW8QdH5zPxRcuO1/IGjqfDcUrD22GkKOrSarMB8UPK96N3xb'
                                                'vZuypwXgjmkBnD30OAiyGh2C740O4bTTBIKmjghDwa7hV2Bg+B'
                                                'WyG2+NtuOtWXPmV6Xl9K9qN/8+KNjqFz0ISu4zCL/rM4gb/1gM'
                                                'gjzyvdAz34sNU7eCoHmJMm+JfvxiQz682pBPzdeC8I9rQRy16T'
                                                'RM2nSatgySfjJIy80RF0GQ3dFwFJzyfhUIKv/QDgUn1QyCxJpB'
                                                'lB/THwUbkr8EQenX3HHdNXdO7rgeBFmgL3ZEX3aJ2AeCHg8OxJ'
                                                '8HB7L53glouHeCSs0heMocwnGVJhDUdCoMBdvUVEDPmgpKb4hA'
                                                'wfVbbsK9LTfJyqMvCm4sGwsvysbS8O2uKDiWV8A0XkGR14fg6O'
                                                'tDONFlB8x02UFJH/rj7A/9uTz1CFSkHqGchcG4e2Ewpzw8C0se'
                                                'nqXqXC1ez9VytsVl2GVxmXJtInCfTQQbcq+DoNJnOhQcP+4OCE'
                                                'qzHICrLQew6UUSGF8k0bxenig41ncTTPPdRGu0Q1FwpuEAbDIc'
                                                'oJKfA1FwdEIJfJxQQto5oRg6J5RzL58HQXZu4djLLZyzPq2ELZ'
                                                '9WUnaIDneG6PjFxVvw/OItiqlQUHBzyl14nXKX8jblQf6mPMpc'
                                                '5EJZi1yg+Ndzbai6uUo1NVcpt/+tBUE2tu+rGtv3laRrL2HWtZ'
                                                'ekiX4/QjP5/bPaYA2GBmvYoB2vJGnHq3mLrPDQIivWN5sUoRZX'
                                                'dMeTFd258EQnENQ+ojcKjtMNBUGXHvVFwXG+E0FQ4W0XFKw9OR'
                                                'v0J2fTCI/B+L7HYHZJXQnOqSsp6TcPFJxetBEEJUZ4o+CgH7Ih'
                                                '5Ids2tV3KAoe0f8AjOp/gIp2+qNgc/ZREFR8LRAFJzmVSF4qIf'
                                                '3lUyAobtI/atKkfxRDhZxTxXWKrFun6uvWKdofnoAgw8P1RqEz'
                                                'Wf4FgvT+GkWoPwR2RMFRL7MVod57Yo2CNd+8VoSacMwWBScPc4'
                                                'GUYS4UdaAPTjjQh9Py3gdB683OKLg64wsQFDR3EAou7W2AM70N'
                                                '9HiOOwq2uZ8O3e6nU9orLxQcGZEFYyKyaPhfPii4bOJeME3cS8'
                                                'k5frgwx4899IfBS3+YbjcFoOAC1xPSo05Q2h9BuOqPIM7oewbW'
                                                '9z1DiYGXQJDe7qwqlIQH9SAoLqyfKhS7D34HQZoA1Amj4Vw7FK'
                                                'yZOFtxmThbLbzbCQXn+FcrQtVpJO9perDVIRuwOGRDmg/tUXDy'
                                                '2mAQ5FbthILN9z8GQQH7B6DgqKp5IGjmg8EouLxiFVyqWCV5zB'
                                                'O/We/JVqd/BEEul7yx/yVvjpy0C0ZN2kVbMoei4KySgyAo4L0A'
                                                'FFxtUQhVFoV0xjMIBVtPPQWCDJYhuMwyhCNvqDDqhkrlt2+BIP'
                                                'N781XTe/OVSM9nIMi0qovRsKqLzjCqFZaPaqWcyQMV0+SBammu'
                                                'JZ7OteS4kwVKwskCtWVINxQcOUcDY+doKKbKDqdU2XHkbTcQZL'
                                                'XXEd/b68g5WaNB0FNjfxSc/DQBUp4mUHmgK14JdOWk75bDrO+W'
                                                'k5WjBwrW/rMeQv5ZT7plQzBi2RDOdd4OgsK/80XBdv/uA0E2Nv'
                                                '4o+MWZAnh1poC8JwXikEmBbDe3GAQljQnGOWOCOWH4Wfhy+FlK'
                                                'bB+KgtOHnwNBGYfNsOmwmVyW/2bUL/9NZ3WsGTofaybNr/t1wr'
                                                'hoZAcUbPNulSJUa/8u2MW/C2cEPFSEWunbE6/69uQEgz0IKo12'
                                                'wNPRkvdUyXtqBL0q7YeCzXNioX5OLJUlDUTBhde+A0EhGW4oOK'
                                                'PjGtjUcQ1leXrhNk8vLijeDEeLN1PKOB/8fpwPp9nlwBq7HMrt'
                                                '7IeCbUYegm4jD5HHrAAUfDv+OAhasyIIBWdllEJmRimZo0PwYX'
                                                'QINzYQPG8gev48FAXbzSsDQWPPvABBGr89YeK06/z/QJDJcoQi'
                                                '1OauVvhHVyuOG2ZUhFo9szsKbln3HjSve49Kn/bCU097ccZoHx'
                                                'Bk3toXH23ty2mJEyA1cQJZZbpgp0wXzhswCwSV/+aKl39z5frr'
                                                'qfDgeiq5kgcOIg926rkRBM208UbBlW93QNXbHeT4py8K9t6zH7'
                                                'z27Kd3Kf4oONHnKAjyPi7nfTyQHc6chN5nTlLUzmCcuDOYC6cb'
                                                '4eh0IxnHhKLgBP15EJR6Tosrz2l5xMzLICixYwsIMsX+qhNG7T'
                                                'kLDDtnwS42WYqDTZZ6x2CNgm3evlSs3r5UUz62xcUf23JjmDO8'
                                                'CHOmyJg+KNh8cxjU3hxGumxnVLKdOef555D9/HMyhA7CpaGDZP'
                                                '8vkf2/hEyKO55T3Lk4KB2KgtLJ6pwXdjon571/K+Tv30qZN3xQ'
                                                'sPe1PeBxbQ/dW+yHgs0V+SAo92YACnZSi0BQQF0QCtaZTkO46T'
                                                'TF7AlBwYmvGGa+Yirw0aLgtM8uwprPLlLuqjDcvyqM9Z4VMNyz'
                                                'gmaNboeC496foQg1dnYn/HR2J/bbeFXx2HhVNey1wWV7bTgpvR'
                                                'vMS+9GuS29cX9Lb87YEwiZewKpdZUT/r3KiQ2uk0GQ3ZcDsNeX'
                                                'A7h0+FwoHD6XRh0cjIK9HVeBoLHTPHHcNE9u7PsjPO/7I7lme+'
                                                'PgbG9+fGknPL20k1pmD8W/Zg/loIiDEBBxkEb84o/v/yJ5Y+sx'
                                                'qNh6jMxdg1Bw/V8lcO+vEgp6FYzBr4LZkKKCoOiNofjRxlDO0F'
                                                '4AQa7/aXHwf1rOGlEO20eU03oIR8FNxZUgyBxkiQ+DLNnw/KAi'
                                                '1IArXVGwzdJWRajzFtrh/IV2bKpxBWONK1VPdsQbk2WfXf0Acq'
                                                '9+QAGL+qPgpMivIDHyKzL8Jef91yCONi0DQa6/uKPgqJj1MD5m'
                                                'Pd2OHoI10UM4KH0bCFoyyRcFR6fsg6iUfRT5yg8FB0UWQEBkAR'
                                                'UHyXsdFMiVXYuhvGsxkV8wCnZ4dgYEtf4agv/8GsKP35jg6RsT'
                                                '3ZylRcGFIy/B0ZGXpP+G4WpTGBe8qQBBC34JR8EvbK6DIOsrnb'
                                                'HLlc6ckXtPScu9p3rf6oFC3u9e0JrYi0ZYOKBgQ30YpNSH0YJv'
                                                '+6Fgs2kq1JumkmvgQBRsHp0MguKnuqHgrJjVsCVmNeW888Q97z'
                                                'y53nEz3HPcTMtdfVCwS/puENT4dCg+ezqUYyvzYGplHv07PgAF'
                                                'B70rBEGxCUEoODWgFH4IKCVXJQQHKyFslUEgSHsnFAW7DimDgU'
                                                'Nkn0WE4YcRYdza5woIspsreXtuOKcnXYO0pGs0a0UECs7R3wBB'
                                                '+s7dETt354QZHSF+Rke6ubEXCraKHgIW0UNo3ri+OHdcX3bJGw'
                                                'dOeeOocZILPpvkwoVrE0GQucQVBRscU2GJYyolp3vgwnQPzk3K'
                                                'gJykDLJ7MgQFJyXsgG8TdtD4al8UrB24H0IG7ieLWH/sEOvPfu'
                                                'oREJSQGYhfZway9ZST0G3KSdKlBqOSGsxxXYwgqDYoFO8Gyfv1'
                                                '5hyseXOOEo9q8f+OajnHTfK222WyaQnDHi1hnHL6KiSfvkp/2k'
                                                'eg4HK6DoIMtjpcaqtj3Qe3QFBza08UnDXUCQQVdeqDgtPeIaS/'
                                                'Q5o1xRkFNw6Lg8fD4ijm74EoOCt0MWSGLiZtJ3cUrF+7FmDtWg'
                                                'r40QsF23XfCoKyDvngtkM+XPDBHhD0bqofCq6Mzofy6HzSFQdg'
                                                'RLG8X9FFIKjAFISHTUE8Iv40CPJbE4L+a6TP7GQQ0se0KFjncB'
                                                'HA4SK1XxCGgtOfXwFBsQfC8dMD4Wx2r4IG9yrSFUagYOOjG3Dm'
                                                '0Q3KyNLhhiwdF7jXQL57DZXs742CW4r9obXYn8aiEwrOSfgIsh'
                                                'I+IoPbABScVzwHCovnkHH2YDw7ezAnJ/wAixJ+oNs+nii4evEm'
                                                'qFq8ierneGPdHG/ODt8JWeE7ye7DodjrQ+nDdADe0QFyuOyPgi'
                                                '28joGl1zGq/j0Qb/weyJptJSCotCYYBbu6qyCo+ftQFJzw8jzM'
                                                'fHmeDC+0uPSFllPal8OS9uU0wjccR/qGc/miSqhcVElRH0TgxA'
                                                '8iuCChGo4mVNMDnQ4FJ9ffkvt7i1JsFVxsq7Dx8R0Q1K2TIwpu'
                                                '+XskCNKE9sd2of258eAMEHSJBqFgO8tl0NNyGT044Y6CY4vWgS'
                                                'AHjyHYx2MIV7ZkgaBFQ3xRcOlfe6Hkr72kr/TDYZV+XHz7MBTd'
                                                'PkxW9oH4nn0gVx88AYJqewfj3d7B7LfhDAiKrwnBGTUhXJxrAk'
                                                'FO/9Ni3/9pufW/i/Dvfxcp6lgYCo4rqQBBM2vDUbB1XRUI+rNO'
                                                '5r0ugl2X3gRB5aTDKyT973gNNB2vIZuVCgpuNdTCO0MtJQ/ph9'
                                                '8N6ccxXT8BQT3/GICCSy8sgMILC8jX1Q0Fp+SnwZL8NIq66omC'
                                                'rQyZ0NmQSbq/vVH5W/JK+91Q1X43PT8/FAXfnpQHgrYPDUDBte'
                                                'sL4d76QmoZG4R/jQ3i27dOwZ1bp0g/OAQFZ+kItuiIpp0LRcEF'
                                                'jy7A0UcXyM4zDHt5hnHs3XKYdrecfL8IR8HFodegNPQarZkfgY'
                                                'KLu92Aom43aMQMHQqOnnUbJs26TZUhCl4LUbg6/icQpHmlYLtX'
                                                'Cpf/excq/r1L43u4oGD9m5kgKH2xKwpu+WIFCEqc6oGCHao2gG'
                                                'PVBko8PQT/7/QQ9qjdDl6120mb74th+b4cvzcXBFWG++PVcH8O'
                                                'GnsEBH27MBAFR70pBkF/zgxGwcknzsKCE2ep0jkUrzqHssuBc+'
                                                'B84Bx579CiYEPDJVjeIL3slzAc9ksYZy6+CoJGdYpAwU5rroPz'
                                                'muuUZ6nDQ5Y6bnG4BYKcHutQsHXYHRBUuF/B4/sVzn5bC9vf1l'
                                                'LreEDBcXF1MD2ujoq/C8ET34Vw4RcMBV8wlSXo8EKCjpsW3QZB'
                                                'QTPLQFDMjAtq1IwLStSo7jhxVHfWqJZgpVqSUe6fKvdvXkolCH'
                                                'og+0VwpUcNVHjUUNkUwItTgBMNdSBo1g5EwU4DzSAoJcoLv4/y'
                                                'Yo+gLSCoeYg/vhniz6V9jkBxnyO03DEEBdvZEwhKOxiGqw7Ke7'
                                                '69AvK3V5DTgb9BkOZsX0WosWU2KDhnQnfImtCdTMXOKNhm+XSw'
                                                'Xj6diq97oGD9hxtBkCb8siJUy60XyGrrBUhY9H0bSuq3qA149+'
                                                'c2lHevTs25V6d4pL0Bt7Q3pLFdr3OxXW/MyIqQfRrBmek3YFP6'
                                                'DYov1OGMQh0/XlwDDYtrKHaXgv/bpXBjfS08r6+lygRAwdlUB9'
                                                'upjux6IfbqhdwU9ABeBz2gsoOIFw8iJ08wg6BiVz2edNWzoW8D'
                                                'pPRtINcNehQcm/szxOT+TJAXgoLTLU2QYWmiBUu0KLg15RK0pF'
                                                'yio13DUXCaZSWssaykirgIFOxRVg2CWg06/Meg45Sbt2HRzdtk'
                                                'tUDBTgsUjh9TC9PH1FLKKMDFo4Bjh9XBtGF1ZH4L+OgtcNOV+y'
                                                'CoNdoX/4725bzUfSAoa5Y/Ck5qPQKzWo9QVvcgFFz+XwlU/FdC'
                                                '88aEoOB0IlhDRPW+WhRs/OIiCLKrCEPBeb2ugqBY9wj81D2CLf'
                                                '68DoK8o3Qo2HXgbXAbeJuKP3DBkx+4cOPIRHg8MpH+XDYYBZft'
                                                '/QEu7f2BEl974jevPVmj2QztNZspu8QHd5b4sOvcPSBobKMfjm'
                                                'uUfhFYAA2Bkjd3St7cKflj4UkQVPssGH96FsyR36gg6F59KApO'
                                                'HV4GgjJqrXBDrfTDjhcVoSYu7YGCNWm2ICjumQMKNmgRBHVW+q'
                                                'PgqGszQNCk3a4oOOX6Clh0fQXZBHhi9wBPzlq7Cbav3UTzbH1w'
                                                'vq0Px360GwQ1feqHgqun5UPVtHyK7HgZBOn1p1ShGN8+AkGa8h'
                                                'qjvrxGF5/3FwjKqbJUMqos1fCB76Fglz5FilALLG1QcEuXLtDa'
                                                'pQt5HbRHwYWFoSBonpczzvdy5kLvz6DA+zMynxyED08O4vpnBh'
                                                'Bkk6IFQfXJO6k2eaeckaEN6ffPbQPNv+1tQ2lG7zbgXX6uDTVl'
                                                'XVeFoo++D8Oj75NfsYPqUuygJG37AwSZnpfqhLF9XEcUnHFzl5'
                                                'J1c5c6YkU3HLmiG5dObw+CGn9ZrTb9slpJ3VVNAjQvFZ3phWIc'
                                                '/uwX0j/7BQyVpYpQkwsvUUrhJcg54gXZR7wo8ed9bSDmn3ltqP'
                                                'eHBnL40ACadjvbkENKWBvQzjZC+GwjVXs2qkLx2FwLgqp7f6Ca'
                                                'en+glIXo8GKIjuddugULLt0iK6MO3zPquPVgDbw7WEMv5P35Vd'
                                                '4f7YyfQDfjJ9LnKTgsT/aB5i481dylchvAKzbAqYvvgSDtHMDQ'
                                                'OcAx9+tg6v068q4AFJw18D5sH3ifhjshCtZOeADhEx5Q61eIgh'
                                                '1mPARBDUcQBTfNMIOguFeIn79CjvzoEYz56BE1eupRcJ5vA+T7'
                                                'NlB9nB7r4vQc/1cDTP+rgTb9qEfBpcU/w5ninyn+vB5nnNdz/e'
                                                'ePQVDOSz3ufqnn2nZPQJAdhmMvDOfqkkqoKqmkgl/D8civ4Wzl'
                                                'ch0Ela2JQMGa/92Af6fekJ2qwzn9dOzwf7dA0NFcHQo2fFQDSz'
                                                '6qobgBCgp2sP8JHO1/ovhMBb/IVLigtBYEJf+r4MJ/Jb+F34Mz'
                                                '4fdofByg4JzddbBrdx1lngEUnPasHtY8q6f6roiCUxwfgKCSqY'
                                                'iCY4MewtSgh5SzB1GwWWsGQT1/RhSc5/kIBCX316Ngb6sGEKSd'
                                                'osewKXq+XdMAd2oayK8oFIcWhXJQwQUQ1LpRi/9s1LL1+UvQ+f'
                                                'wl0saFoWDXnVdA0CjncBRcPKwSSoZVUuK1cPzmWjgXnqiCohNV'
                                                'ZJ4VgY9mRXDln9VQ8Wc1GTvoUO2g48Rut0DQu3QdCq785zYISu'
                                                '2i4MouCtdfvgP3Lt8hnUHBCIPCibNrYebsWqr/VcH7v0re+/Mu'
                                                '/PvnXfKYAChY83kdCDLlA57LB847Vg8HjtXTiP8ABbc+vg+CdO'
                                                'MQBT9u/xCetn9IAZsRBXt0MYNXFzNVTw9CwTH9S2Fq/1JKWBmM'
                                                'X60M5sTuRhDUbWMICk44yfDlSaacVaG4e1Uoz+t9AQSVz9Si4P'
                                                'jQSyCoty4MBVc7XYEqpyvU0hqGb1vDuLniKggyHg3Hs0fD+fbI'
                                                'KhCUNSkCt06KYO+N1eC1sZrynkXgwWcRnFh4EwS9mKdDweYDt+'
                                                'HBgdv08R86FJy88A4sWniHyv5PwYv/J/M2uBZ6D64lzV0F/6tV'
                                                'eETRXYgsuksmRX4fBThlQB0sGSB5aTvgie3y/s2rB0GNvfzwWS'
                                                '8/9v6/QyAofYM/rt3gz3mJRyE/8SiN+TUABRf2OAEFPU5Q+pAg'
                                                'FDw27hSMjztFhknBuHSS9NtpZ0GQ1/QQFBzfg0FQYmwoCs5Ych'
                                                '42LTlP74ZrUXB5/kUQNNM+DIXkoXLJQ+XkXReGPnVhnDbpKgja'
                                                'siEcBSfcuQYz71wjc3AEPgyOYDuvaujpVU3e1yNQsMPEm9B74k'
                                                '1KjdXhilgdF4+/DUXjb9PtBzoU3Dz4DrwefIcspijYYYrCxXd/'
                                                'gpK7P1HrEi/8Z4kXGyu3gKDqr73xxtfe3Nx/Jwi64+OLgrPb7Y'
                                                'Pt7faRXZX0waqhnJacB2uS8yh+nD/OGOfP89YeAUFxRwNQcJm+'
                                                'CM7ri2jem0Cc+yaQc3aVwN5dJXTTORgF1+efgQf5Z2hTQAgK2S'
                                                'cEjv9HlOkXioIzHp+D9Y/PUZajFrc6atnP6SIEOF0kv9daFFzQ'
                                                'uxwEPSgNQ8FxtdKnaiuoZXY4Ck6YcQ2+nHGNjA4RKDj93HUQFF'
                                                'UcgROKIzi+6QZ82SR5TzcIv9AN4pybS0BQ5unB+OPpwVwevQou'
                                                'Ra8io4UHnrXwYO/S9SBo7wAvFJy9fTNkbd9Mtbbe+JOtN7t0yI'
                                                'aBHbLpwAMfFFzbsgcEea8cioKzlx+EnOUHSd/ZH4d19me/gwUg'
                                                'KD0pANcmBXBr4nH4N/E4Gc4E4tIzgay/dxLg3kmKex6Egs3Np0'
                                                'HQiPYh+H77EE4+r4IgO4tQFBwXeQ6mR56jLb+HopD8UQZb1pZR'
                                                'QqUWv67U8u2My3Az4zJZbwnDrlvC2BBbActjK0hT0wfb1fRhzf'
                                                'kR0HpuBPl17YdDu/bjvI9iIPejGCrr6IKCkw7PBEH6CwNx2IWB'
                                                'nFXzPQh6NXowCs77bSUc+G0l1e90x7qd7nKf1oHX/62jB+c9UX'
                                                'BSfCYIyjAOQcHpDdthTcN2ilzvg4LTX+TAmhc55BExFD0jhnLc'
                                                '6gMQv/oAOVzywz6X/LiSD4Ogcv8AFBy1pRDGbymkYoPkPUMg+9'
                                                'meBF/bk+R6MggFR+tOw8e606S7EYwRN4I5xV4Fg71KjXdC8Nmd'
                                                'ELbIN4GgP093RcGam38rQm2p64F/1fXglt29QJPTi/yO9UL/Y7'
                                                '04qZ03zGvnTSnaPvi9tg/n2Q4DQQUpTig44WE0xD+MpiRDf5xt'
                                                '6M8es78CQVGRA3FC5EDWXE8Gi+vJlF3rioJbV6aC5odUmhfsjn'
                                                'OD3bmxTzo87ZNOhdM9UbDO60cI9/qR3n0xBAW70jYQVKTzQcF5'
                                                '9bsht343WTX5YqcmXy5P3w+CXhj88FeDH4+oyYfhNflk99ofbV'
                                                '/7c6HxGBw1HiOrUYEoOGZ0MQhK29sKgkwubkq1i5uauq4DrljX'
                                                'gTUd1ypCtUu0QsHVk0xK+SSTmvC7NQo2/fZaKf3ttaod2AMFa7'
                                                '/pAYJiOvTCTzr0Yj9rD/C29qDCbQ54bJsDZ0xVIHOqQtbP+6Jg'
                                                'U9xEMMZNJNe/nHHwX87c6BEPgtZcHYCCG2k+CNo+wxUFV9uvgC'
                                                'r7FZR81Q2/u+rGY8vWwJiyNWTTzhN7tPPk5l83giCbzkOwe+ch'
                                                'bN6cBYIeP/NGwR5XdoHblV1Uvs8Xr+zz5bL0XPm9csnO5Q4Iam'
                                                'qcpmqeTVOinH6G8U4/k6Y7G4XuxQe/gSDNynsRhtR7Z63z3oEg'
                                                'lzE9FJsxPdSiPe1RcFPPZUpjz2Xq84z3UHBhfYmSV1+i2rlZo6'
                                                '2bNVf/+Uyp/fOZWjvDBu/OsGHt0K4QNLQrdZtsh4Kru7jC7S6u'
                                                'lNlij5tbJL9vCQNBoyb3RcE5P42D3J/GUf1sZ6yb7cxZtXEg6N'
                                                'XYASjYb/+34L3/W3rcNAh/bhrEFvuXQfv9y6jxMzd89pkbR89c'
                                                'DZNmrqZ3Oz1QcNqhDFhzKIMqJx4EQabQQW2g5WUJvHtZQk0/at'
                                                'pAdunFNtQSpaqaiapidewWCCr8doFq+naBknxDeu4NM2kyXxhN'
                                                'P77QpQ98CYI0zQXhmtcFZ5wOtoAgzcYWnSmjxXipqB0Kduk1S/'
                                                'HrNUutLLJEwU0TjihCbZrSBX+f0oX1v5gV7S9m9faB7lhzoDs3'
                                                'PbECQZG5tvhBri0naVwgUeNCXpOl70y2Z015MLReDqZocsSPyJ'
                                                'E1I8aAoPTH/XDd437sseRTEJRU6YJzKl24dNlsEJQw+HOIH/w5'
                                                'NXdb0wb0GcvbUNLXSW3AXPcjPKj7kTxujW0DGV/ltiG/DZ5twM'
                                                'OzuE1bP2sDNo4XoJvjBUpYfEWNW3xFsQi9AYLiuq9So7qvUkqX'
                                                'PgBBLgFdVaEk/fkcBBmiVp7VTFwZkbv7TxBkCK3SCWOpSYOCNb'
                                                'WfKEItvNERBRcW7FPyCvapLT90xrc/dGaz6SdFqAOfdEPBkRst'
                                                'QFBrU0/8u6kn1z7rC7ef9aW4Y73x82O9OeaVP8S+8idzYpVSm1'
                                                'ilOnQ6TwI067u1odzjhW2gtEtwG7Kx290GClOmtqGSMeuodMw6'
                                                'KCxe0oZ0jxaQ/tECiL6c0Yayhn3UBgoW7GlDzc+HUtPzoZDy6f'
                                                'E25NK9Kzl07woZ80ywfp6JrEprVE1pjWJRWgWCzH9sUoXi4F4P'
                                                'glzQVRVK1oVGEKTf0Neo2dBXl7OxGQQZVu/TmVbvM1pv/Q8Euc'
                                                'weqQg146UFbnxpwebX2xShup7ohILNK6oUocadumw0nLqse1HQ'
                                                'QAIM+5N1+v3JxoC5T8lv7lMwVXymCLXE6g6VWt2BOAuTIlRfV8'
                                                'kirhch8vv32pBX/QnyqD8BCa6+EO/qS9NH76X40Xsh68pHbejP'
                                                'lRnUvDID9EpyG6r+ZDGVf7IYMgrWwfqCdTSv6BNKKvoEvJfuak'
                                                'OVY0KofEwINK852oaCJtuS32RbsC5X21CepkHN0DQoFgGVICjv'
                                                'xS4158UuxXD5LghqSopQzUkRSvPqpyBIbzffqLGbr7NY/BqE7L'
                                                'c5OkO3OUbvbfbksc0egnYWkADzA7MqlDHvEUW+R9Dos1E1+2xU'
                                                'Guyvk9n+Oph926lCST71kARo/Dvo9P4djG7TnpHHtGdg3jRcEW'
                                                'p767uksb4Lmk3HFatNx9V3EeXUElEOpZ9r2pDO4lQbyAlyh+wg'
                                                'd7JbvL8NJFmPg3nW48hg+pFSTT9Cxpk5bejjZ8so+tkyyFq7pg'
                                                '3NGxxHSYPjwGPZjjYUfVRHMUd1UHD8cBsy5PRpA7l2Z9tQ8sev'
                                                '1KSPXyk2+6+AIIc3+arNm3yl1N6WBeZddQJB71e68YhKN7x9aQ'
                                                '0IyukWzgIbO1XC806VtPGowhlHFTTb34UH9nfp1cZH1LTxEeQU'
                                                '3DeaCu7rloztwALNI9MUob7+vQcLjOvfGwTd/7gf13/cDw32Uy'
                                                'HFfiqdsO/Pxfb9UfM0HiyextPcBe48b4E7Vv6aDhW/ppOl2YcF'
                                                'JvyzBwQdvhzABZcDMHlrESzaWkQ3VwWywIIhJ+HokJOEfUNZ3z'
                                                'cU87adgwPbztEHtWEcWRuGQZFXISTyKlX9HsECb1+7CYJghI4F'
                                                'Or2VPv/2FtVfky56TUHj6rtAq+/S8lZggY/v3QdBc58gz3uCmB'
                                                'DwCATFlRxvA/WR1lQbaQ0xVVdIQItfvtrkl6+MPfmQBOhHtFOF'
                                                'kvDtGxJgmpiuE8Zr69pz5br2mCPFN0tnUFcldOa0hM6YlHhHSU'
                                                '68o66P7cECY6gnRFNPWrXEntOW2GMchYCgrIt/tgHTuBu6nHE3'
                                                'jEM6WbB3JwvM0GcoQm0u7cICCz2fKKWeT9SClT1ZoMd3fUCQY6'
                                                'oDC9Tf1YEgmyv9WKBNTSxY18TS3oSBLNCw/TtI3v4d7Znvxjnz'
                                                '3dCpcDU4F66mfnm27JRni+beLiDo2Io+XLiiD2pqh4NF7XBaeM'
                                                'iZkw85Y5RuOgj62GcQC0yeuwQEPRjgzgKN99bCmXtrySbLiwXa'
                                                '9N8K3fpvpTG7fVhg5OA9MGrwHhoxUrLdSD+M6pIP47vk09cvB3'
                                                'HCy0Go2b4M/t22jDpWuLNFhTtav1sHgvyHDGG/IUOwpf02EPS2'
                                                'vy+39PfFoFt7IeTWXqo8Kdn6pB8a0g5DatphyR8BLHAEngBBFa'
                                                '+DWKB1zzPQuecZ2nkkhLOPhGBsHxMIytzpywJbF+bCu4W5tH+Q'
                                                'P+cO8sfkfwpg0T8FtD8mkHNjAjFmteTN1cV0Y1QwV48KxvLgs1'
                                                'AZfJYGvQ1h17chqOl7Dv51PEdz52h53hwttoy/BIJ2GMM42xiG'
                                                'hU8qQNDmB+Gc+SAcnX6vAuffq6guS7pkVjC6fGSEgR8Z6TmEss'
                                                'DHvc/D097n6YNCLUcWSmf0uQyCst/I33sThlmHrsL2Q1fpfLcI'
                                                'Fli+8zqU7bxOVyx1XG6pw4Q+t2Bmn1s00KxjgXledyDf6w7l7l'
                                                'JYYNP9WhA0yTOcBbp+XQkuX1fSVSWCK5UITB1VDYL+G6pjjZ8O'
                                                'rY/egs5Hb5HeQmGBqUfuwPIjd2gCKRxFCrpOvAsDJ96luZ8Dz/'
                                                'scsHxfHVzaV0drm4DTmwCtj92Hbsfuk90sZIHzvn8Ign5wV1hg'
                                                'gs9PIEh3X2GBpefugqAfU4AzUwDrLevhnmU9fWyFLFBn+QDCLR'
                                                '9Q83Jkga7lD0HQulfI6a8Qmz56BILajdOzZpxe5r0BHAsb6Plu'
                                                'PQuMafgZBCXgljbQWvUBtVR9AIaCfSQg3XUIZbgOAcfMQnLIlB'
                                                'xxvRsJmPXaSEmvjaC/+UQVivfFchKQ1+2YWtjtmBLb5TYJML/7'
                                                'VhXK2IsPSEBSamfVkNpZ+bK5kRKaG0H/wNZoeGCre7X9NTVtfw'
                                                '0G83yd5tF8o4XxbxJg2uOkFO5xUot92rNAlxnJis2MZPUIWHIB'
                                                'WGLT5DylcXKeOty2Mws0n7ipCPXFka4s0G/034pQtT2k2/TogV'
                                                'ZdeoB1lx50Ae24DO2waf0gaFw/iBqcmMxODC5P6lShxN2pIAG1'
                                                'k/erpsn7legOd0hA9b+fqeZ/P1O8c80kIMeuyWiya9KlnnlOAj'
                                                'Rjss6KiOrIZhKg379T57J/pzFy0j8kwPS5j5LzuY/65cv2LNDm'
                                                '/VRFqDFvLFlgVK9CJaZXoRqtdmaBcQPuKVED7qm147qxQG1AOw'
                                                'gKaEcLD/bg5IM9sHy2HVTOtqNrlXZcWWmH2qtuIGixyZ5TTPZY'
                                                '/oEWTB9oyXmlIwssfBYJgkb068cC9SUxoC2JoYwlDW1AP/GqUe'
                                                'h0KS9IgCZt+2mNYXtY/Os/SIB+kEnnMshkjH/8LwkwdAQlqSOo'
                                                'vQ53YLvDHdBmZLriMDJdtT3wHtsdeA81l04pNpdOqatndOG0GV'
                                                '2waekjpXHpI/XfX7uxwORSC0gptaCOup5soeuJftfsQRDE9GKB'
                                                'URovEDTqfQcW6Dc5ArwnR1Dpf44sMKb6QxA0MKMfC3RZ+D8Q9N'
                                                '+f/Vnztj9mTZgJgkpDBrLAmLhkiI1Lpt0TXTlnoivqa6Un1C6n'
                                                'IWEa9g7TYE76eMWUPl698KUFl31pgUnaTCVZm6k++NiKBTZ9xU'
                                                'rLV6w+7WLNAk3XGxWhTk/rzgJjQq0gNtSKnlb1ZIFRg/pC9KC+'
                                                '1KuhF9s19EIPXx/w9vWhdWUOnF7mgFn/AxD0NLkvC/QoHA+uhe'
                                                'PpUDtnzmvnjDH+0yDWfxpN+NyFoz53wYQniRD/JJFw+0DWbx+I'
                                                'VrcWgaCyo64ssECTCnmaVLLb48YCb0dI/olYQx/M9eDIuR5Y6Z'
                                                'wBVc4Z9Lm7F8e5e2Hikc2QcGQzzSq0ZoEmmyal2qZJ7TjIhi0G'
                                                '2WDt0S5Qf7QL9Y+zZZc4W/T41hm8v3WmqTN7s8DGaD8QFDOiDw'
                                                'tsmjoMmqcOo8NNfbmgqS86rJ4IgurmOHP9HGdseRwHguIqXFhg'
                                                '7bzZIGh760AWmIaLIR0X07XOg7my82B0cFsJvd1Wkv9bN/Z764'
                                                'YuE9eC88S19NMFD6694IG5/0/R3YfVfL9xAD8jhBCikCQhoZJK'
                                                'D+d736czIw+jEWvE8jCamYXQzMMZIcSSbI0QQjOzTD+azve+Q2'
                                                'iVhBA9OCmVOraGEcLv7o/XH11c19f7+/l874eL6+K8C44576Lz'
                                                'y1xZ4JrUH8GQ+iMZ2wxjgdHF+2Bl8T46vG44CwzPTIGwzBTq9c'
                                                'CN7R64YfXUo/Bw6lGy/rcnCzREjQRBaRd7scDQqaNBUMeF9my1'
                                                '0B51M0Ng9MwQIpODFAzJs2seCBrp2Z+9PPuj6e1SKHu7lP6aNY'
                                                'BzZw3AuqProfroerrwxUDO/mIgJgZtBkHjJw5mgcnz4kBQZPch'
                                                'LLBgcAIUDk4gN9WVBVr8/BMImjVhGIdNGIYWWcnQNiuZLvBwzu'
                                                'bhmLLrEBzedYje9XFngXGOx2Cb4zH6ZY4Hp83xwPlfnYAFX50g'
                                                'z4QRLDBjxSn4Y8UpenvKk5tPeWKZ/g8QNHVSPw6Z1A8jfBfAYt'
                                                '8FdC2pPxck9cckQxQIuqEO4CJ1AFq3/h4EpfBAFugcsQUEzUoZ'
                                                'zGEpg3H08h0wZvkOsp07hO3mDsEgl90wxmU3UduhLPD5tiTZg5'
                                                'LINX4YC3Q+uB8E9dS4scDEiMPwY8Rhcgl1Z4GahmPwrv4YmX/2'
                                                'YIEa11/h3ZBf6fqVEVx0ZQRaDfkdBI00ebKXyROTG/4AQZ/Vju'
                                                'TQ2pFYHXUWBK0s92KB1rl/QrfcP+n4JW9Ou+SNw+qM4FpnpMV2'
                                                'g1hgbkwsCFr6bDBHPhuMLoadIKjg9yEs0GZQIgj6avJQXjx5KI'
                                                'au/hlmrP6ZnIqHscBiwwEQ9Au6cRq6oU45AnrlCO3b7c7Ju91x'
                                                '9JnjMObMcfr+hgcbbnhgTI3MazW/UsHrESww6v7vsPL+71RkPZ'
                                                'IFRiecgdUJZ6jIxosFFnU9B4Vdz1FzO28WUj/OS/04T4sbvFlg'
                                                'wTIVBA3J9mGXbB8MnpENk2Zk05Ato9hlyygssrkEgqYF+nJIoM'
                                                'xnqZfhdeplCvhpKGt/Goq6hXtB0ATb4TzRdjg6hx8El/CDtGqd'
                                                'G0evc8PULqlwrEsqfVrozqGF7hjxbRoI+sZyBEdajkDNoZMgyM'
                                                'XNkwWat6aDoHu6kVyiG4kxozJg86gMMgZ6scDmk+fg3clz9IGn'
                                                'N2s8vTH6yXlY/eQ8ZXbzYYHVb1QQVF/twwLtS7PBoTSbdvwyiu'
                                                'N+GYUlOy7BnR2XaMMcXxaY0vMKHOh5hQ508mOBdatyQdDRU36c'
                                                'esoPLTLyoFVGHk3/yJ8FOl4rAKdrBTTL0oPDLD0wXPkFZiu/0B'
                                                'kYwRkg8+9nv8HKz36jx+GeXBfuiWtGnYboUafp48iRPDFypMz7'
                                                'GSCo3xIvdlzihdFjMkHQhpneLNC4NgvOr82iHX4+HOfng9EbCQ'
                                                'RB+1Es0DDjAgjakj+KY/NH4Zo2OSBo4ve+LFCz5QoImjnUj8OG'
                                                '+uHiylwQtCbPj9e0/BsN23zoaZtPcz/3Z4HOrtdAUFa9Pxvr/T'
                                                'HV/jocs79OLl8HsMD4xiJIaCwiv9oA9q0NwPTjNyHj+E0acXAk'
                                                'exwcicPW/A8EvfnZi5t/9kJdRabMaJm0JcabY2O8McLKCPOtZD'
                                                '7/XM7jcx+Mbc8gyGn4KBYYVXIBBNk/GcUCi77LgcLvcigjxZcF'
                                                '2r+4Ar1fXKHsID8WmDPuL7gy7i96+8iPmx/5ofW3+WD1bT45fO'
                                                'vPAu03XwNBN1oFcFGrAGxadh0E/bExgAXGB96AhMAbNK05gEOa'
                                                'A9D47CYICvxay7qvtei8vRhcthfTiztaFhjV/g6sbH+HGv0UFj'
                                                'gs8i4IKt7kwwLtdjL03smUFDyKBWqmXYR3IRcpqLPsH519Mbcx'
                                                'BwQx+0r99cXn865C07yrlP+FHxd84YeN6X/B3+l/kcsH/iyw+l'
                                                '4+CNqR4M9xCf5oUXkNBB3sHcApvQMwOPc6TMq9Tq9+DuCmnwPw'
                                                '5K4bcGLXDXLvqmWPrlqsg1tQA7foskHLOQYtpt8oBkHKYy1rH2'
                                                'sxdcIdEGQ1QWGBo0/dhTGn7tLBYwqnHFOw+XUJCJrTrHB4s4KO'
                                                'I++D88j79PtE4PSJgM8/LYUXn5bSogpfFphUdBX2Fl2lNRvkfm'
                                                '3wQ3PPPPi7Zx61tvdnC3t/TPMrgGN+BXTqV38WWOdfCPX+hTTQ'
                                                'M4CdPeU+2RfBKfsievt7ADf/HoBlphsgKG+glgV6bb0Fgrokat'
                                                'k6UYuLbW/DItvbNPGNlgVa77gD3XbcoeczFRaY1nAXBK06q3D0'
                                                'WQUTve6BIJ+OwAJdvrwPrl/ep10zgONnAKZuK4Vj20rpUSpwdS'
                                                'qg794y8NlbRoZ6YIHPk8qhMamcMoYiCzRtqQBBG3P8OSbHH3N2'
                                                'F8KV3YU0d1wAC7SPLILekUVElwNYoLP7TXByv0mD/bXs4q/Fkz'
                                                'dvgSDH41oWGBZ2G2aE3aZBnRQWmH3tDly8dodOLlZYYKhLCcxw'
                                                'KaE+VxS2vyL749f3QFBZL2CBiQfug6AJEcATIwCzjaVw0VhKdB'
                                                'pYoNdfZSBo2wtggZaXy6Ht5XJ644Pc7CP74OkKgNMV9Oly5NDl'
                                                'iLk7H8CVnQ+o6lfk6l8Rk2abYO9sE5WakMtMiIb+lbCufyWNma'
                                                'RlgUF9imFin2J6+aeWm/7UYkzKbdiQcpv+clA410H20U534Wyn'
                                                'u+S+VmGPtZJnfgkISritsMCCo/eg8Og9mu4CLDD81n0QVLpC8q'
                                                '0AnPh3KQhaRsACT74sg1Mvy2hpK+TIVogpjeUgyFGPLNDxfgU4'
                                                '3K+gq2uRc9fKfvu/B+D0vweU9j9kgXEbTLBzg4ks6pEFaj+sBP'
                                                '8PK8m6j44Fml9Wwt8vK6nNOB1bjNNhxKGHsODQQ+oWpWOBITrZ'
                                                'f3VV9P0OhQ075L4dLIETB0uotFrhsmoFI8vvwbLye3TcBzjNBz'
                                                'CpXSkktisl/UZggaP7lYGgD/KBNfmAyQPL4cDAcvLsjCwwo08F'
                                                'CBrwMbLzx4ixrR7A5lYPyC4WWaBl6QPoUPqApjFyCEveYyZwOG'
                                                'airs+QrZ8hxs+vBEHbB+g4boAOPWwfgqftQ3oRrGOBRnoIgjZ9'
                                                'p+OY73RYNrMKBM0/omOBXv9Ugc8/VXQxV8dC+nW19Otqutyg45'
                                                'wGHWreVMO719W0O0TLiSHy/XkWQ2fPYvrolp5H39JjRlcznO1q'
                                                'poRtcr7bFNTsKYF3iSVUfk/PZff0GOFqhkWuZpoU6cECNftPwL'
                                                'tkmTcn61igV+ZDEJS3W88CQ50bYLpzA1k/1rPA4HlmmDTPTF0U'
                                                'T7ZWZB41poOg5h90LDD6eBUI+mi//Hn26zF7bANcHNtAq+r1HF'
                                                '2vx7gIM+yMMNN3ZSN5TdlIjJx8FhZPPkuJeToW6GyoBkH3juq5'
                                                '5Kgeixc2gKCsBj0bG/QY9LUZBPWc6s0CF4dnwZLwLEp4r2OBIa'
                                                'MfwfTRj+jAb3oWWB3TAKaYBjKZ9SwwcakZBBXcG8ACc2Z+D4K+'
                                                'mTKKI6eMwuiZF2HlzIvUeiewxU7AuDFlsHNMGQ0PCuRhQYFo+K'
                                                'QGNnxSQ8FT9CywyL0eCt3raXSm5M2U93+yAc6ebKAvH+g54oEe'
                                                'ndEMg9BMy//Wc9TfeoxaZYaVq8wUFjGQBdYFbwZBrWpGscCcuT'
                                                'lwZW4OTbkMLLDEuhzuWMv9nBXIAmMO1oCgGTP0LLDo43q4+XE9'
                                                'zciSn7P0qM1sAMhsoJ9Nek4y6THlQzMkf2imWnl+nTzf5VszCG'
                                                'p6OIgFNnXZDi+6bKehy3x52DJfjPnkiuS7QnWvgAWG3SiHGTfK'
                                                'yWNpIAvMflwDgj6Yq2fNXD2GLaoHQeNIz0GkR99LDeB/qYH+rN'
                                                'RzZqWc31gz3Bxrpun/6FlgwWoz5K020+7xLpw43gVP7v0BTu39'
                                                'gYJf+rLAYpdcuOOSS4+k3ldLvU+MrQBBI2MC2SsmEMM9aiHMo5'
                                                'bGfinP+1KPsZvqYdumeup5Qc8CT15rAEEFD/UssHiCGe5MMBPL'
                                                '87Pl+SFrzCDIIWUICyx6uhtuPt1Nf63w49wVMt88/gsEfTUTef'
                                                'FMxOc+D0DQuB8DOejHQIxfXguJy2tpZKSevSL1OP9gPQiiS3oW'
                                                'mHK3AQ7cbaCiKj0LTJ1sBkHOjXoWWLTWDIVrzbSi2pWjql3RuX'
                                                'cSCHryxI/NT/wwdE8+zNiTT8lSbwSayx5AfdkDWnc8kAXGnqkF'
                                                'QfkrJd9KPWoy60HQR1fk/l3Ro11lA9hUNlBetXy/1fK9TjXD+K'
                                                'lmipHnC/RabwZBK+yGcZTdMFzcLRkiuyVT13B/tg73x8Rh10BQ'
                                                'whlkgWWrTVCx2kRl5wJZYHhjLcxurCXrtVIP1ko9uV4Pi67X08'
                                                'lcPQu0qZfn1zfQ2Ud6Fpg93QyCHsjzTfL8WIMZBD3RDmezdjhG'
                                                'lB8EQYYCfxYYnl4Ic9ML6asyef9liMO6VoKgxVcDWWCQax2Mca'
                                                '2jthv1LDD9UT2celRPL/L0LDDlaQMcftpAP9XI/a/Ro9UMMwjy'
                                                '/lfO6195/9+boeB7M5080psFJh0eB8mHx9Gl6a6cM90VjUE/Ag'
                                                'X9SGmyjwp8vuIsCPrmTz+O/NMP59/PA0FWY2V+G6tgzsG7cPHg'
                                                'XSqNkv4chRiW+AAEDS7QsUuBDps3VUPTpmp6dTCQmw7K/fmpFh'
                                                'J+qqUaJz0LfL7uMQhqWK5n83I96n6rB/1vcp+PyX0+psfMiAbI'
                                                'iGgg18t6Fvi8rAEayxro6xI9Ly7Ro8VAMwiaJecdJued8YkZBF'
                                                '2Sepkj9TJC6uUiqZcz5P0LTFpnhh/XmelgWh9OSeuD4Tcnw/yb'
                                                'k+nzF64c/sIVbaYlQbdpSfR7kBenB3mh5vI5aHX5HMU/92Mh+1'
                                                '++7H/5BF8pLHDYgBJwGyDz0XZk++2IdjUPQEi/0Um/0WFGZjWc'
                                                'zaympWmBHJkWiNbnaqHbuVoa4CLfg4seDQmPYV3CY3JZpWeBoX'
                                                '/Ww4w/68kuTc8CfSMbQNCXcr8j5H4ny/3eK/d7pPQzL+lndYPN'
                                                'IOiO5Be4Ru77Ornv1lLvBcYsMcPmJWaKlfwCg+T+j5b7b3vcnu'
                                                '2O22P6w2kgqGbXUBaYNm0vCNIe92KBLlZ/giBPN38WGLu6AATF'
                                                '7lRYYPiREph7pIQKUpAFFnmbQJC2VscCfeurQdCZ9EDOSA/E3N'
                                                'u1cOV2LemG61lg3NHHIGjqaj2HrNajx+V6cLtcT51O6NnqhPz6'
                                                'qgYQdPCqnlOuSv2uaQDnmgZ6KvkFpks/z5B+Plm+t2D53rTTzK'
                                                'CbJuct+QUmfWMGQZmSX+Bzyf9C8gcc7Mvag33RdGsmVN+aSSuH'
                                                'DGOBxvHJIOjz1t4c3tobEyecB0GfL/Dn8AX+aPfhNej94TX6J1'
                                                '32sXQFox3vwWrHe/SvzHeNMt8lbjTBjxtNNPa5joOe69DQ+xEI'
                                                'upgZyAKzn9XKvldLWk89C3TOfAyCFko9iZB6klFUD38Uyff8q3'
                                                'zPv0q+dQ0gqFDqi8BEqS8JUl/S7+tZ4JrhZlg9XO675M+R/Gap'
                                                'N4IOSf4UyZ8WaYZjkWaqlvwCk6X+CBq+y4GH7XLAoGNzQJCLOo'
                                                'wFGrwOwDqvA+Qyy5sFRm3MgmUbs8hxvz8LLG5bCIJW3ZT95qaC'
                                                'hr2yD+y9R/F5yAKjr5sg6rqJ3mkCWWDyuEdwYJzMI9mBLLDZug'
                                                '5eW9dRz1HyfY+S88p7DLq8x7TIoGeBjaX18LS0noplfhF4cpP0'
                                                's00N9OIveR9/yTzzdwMIWlCqZ4HDPGS+8JD6JvXOS+pdcqjkCz'
                                                'VTneQXmC7zjiBrqX8CdVL/QOqfzdp+LDB4xkIQ1P3j4Wzz8XD0'
                                                '6J4Cgk5keLPA52eMIGjtTX9ec9MfSy5L/suF5PpMYYGJXWX/6X'
                                                'qfTlUgC1zTqxJW96qkBZ0CWcj9fQTxqx5RUK70z9xAjB5eJ+dV'
                                                'R7cDJF9AS97HIGim1PMwqed1NfVQX1NPR3/Xc+rveoyOa4CVcQ'
                                                '1kmy/1IF+Pjs8awOFZA/Uqk5/LpL94muGip5lSJb9ArdR7QcOf'
                                                '6HnYE8m/TPIvM5O/5BdokvwVkr/zPEcWGNS4GCY2Lqbge8NZoK'
                                                'b2EAhqaO/D5vY+6FWugqCYtgEs0MtwHXwN18mqO7DA7E33QdC4'
                                                'p8hBTxFPzqmE9DmVtK6X9OtegRiR+ggWpD6iC9cDOft6IE4cVw'
                                                'fjx9VRlE7mQZ18708eg+eTxzRqs559N8v3+m897P23nr46LfX9'
                                                'tPTz3Q0gaESBnj0KpB6+lPnqZQPlSn6BE71lHveW862V862V+W'
                                                'qmzBczzfSN5I+U/CeXm0HQPMk/X/Kv2SDfywYzffhhfx79YX9M'
                                                'i1gG6RHL6OUcN26a44bO6UdgUPoRajPLhy1m+eD89wQL3hP5+g'
                                                'WwQBufIujpU0ROnsAC7V7eB0FfttFxRBsdOqZVglNaJY0fGMgC'
                                                'HYsegUPRI+p4N5Ct7gZi5Pw6WDK/jppG61mgxbvHIKj1Vj1bbJ'
                                                'V+3VwPggaekf5wRurfzw2w8+cG+vCazDfX5HzfNICg5eXy/sql'
                                                'n48yg6DVkl9g2CwzhM4y02nJny75U6LMcDjKTDskf5zkt95ohm'
                                                '4bzVTc14kFllxaBYIca91YYMmSo3B/yVHSnPJhgZm9s0HQ8UUB'
                                                'nLYoABNriyChtoi+CQaODJZ9+ItSOPFFKXWxk/3VTocl5kq4b6'
                                                '6kfI9ALvAIxJzXj+DK60c0pCKQXSoCsXpdHTxcV0ezxsl9H6fH'
                                                'xZ3rYUnnesraLvvNdjnftnK+bRvobobsPxl6LDso+8vBBppcKP'
                                                'W9UOr/+wYQZFsh979Cfr+/Gfz9Zb6S/CbJXzLbDIIaJb/A+BVm'
                                                'EJQh+QWmSf5jkr/qqRNXP3VC31ZrQdDDCHcWGDzgOAgyN8v9b/'
                                                'bB0e4XQNCdvQEsMDvxBlxMvEGvvwYWGHujFAQprtLvXHVocn8I'
                                                'giy0gSxwvnMNCLJ9FMh2jwIxI6kOBB3/WM9pH+sx0kHyO9TTvJ'
                                                '1yP3dKPmvJZ91Aa87qWaDlsQYQFHJd+uN1PTa1MoOg85JfYJJW'
                                                '+pvWTKPr5H7UyXwfboYp4WZyk33HrWXfWmmG6JVmKpX8ZZLfN8'
                                                'YMguIvDmCBdt2+B0FTat1ZoPFaGpy/lka3x43i4nGj0Fe5CP7K'
                                                'Rcr/K4AL/gpAo3ITzis36a9twLnbAHP8y+CKfxk1BehY4OilD0'
                                                'G/9CH9NyaQn4+Rej+xBgTdbwhkgdWn5fxP11HJJ3K+n8h9dKsH'
                                                'QW/i9dwcL/tKrwa400vOV/ZJO9knrWSf7CD75NQiyV8k/bGt9P'
                                                'u2ZgqV/VJgHcj8A2Y6JvkF5syReXuOmb6S/Islf5jsm4I0T2Vf'
                                                'eyr1UvJflPxv45y5Oc4Z7fdsBEGb53qwwJAlJ2DKkhOUmziKBR'
                                                'pHXwIafYn2vAzgxJcBmPLwJhx4eJMqjgELjDxUBoL0H+tYoM3p'
                                                'h2B9+iGlfBLIApOiauDHqBpa/jSQo54GYlReHUTm1dGTaTLvTp'
                                                'P6pdSDoA279SwwaEADCEr7U+7Hn9LvTku/O91AE27oeeINOd8O'
                                                'sr91kPst+QW6BJrBOdBMbR/LPvBY5ol50g/nmSlN8gvURks/iD'
                                                'bTIMkvcOImMwgaP2UgCwzrsBkEJd73YIFBbU7C+DYn6U75KBbY'
                                                '9FEOCPJy1rLAmI23YMPGWxR4EVh3ETC5TTkcaFNO/Wbr2HG2Di'
                                                'f+8xAEtQ8LZMuwQPTaVwOe+2roYVMgC1z8sA4WPayj+s/0LHDY'
                                                'xHpwnSjnv0fOf4/Mw8MaYMOwBnKXfd5D9vnkczLvnmsg5abMSz'
                                                'elf3UyQ1knM22R/T5W9vsQ2e9DZb+fL/kFxsw3w4b5ZqqU/CbJ'
                                                'by37fjfZ94Mkv8AiyX9T8v/YbRALtF4WC52XxdLOySNYYFHCby'
                                                'Co7UBfFmgZeBnaBl6ms5O1LHCNYzEI+q8M+HkZYOjCcpi+sJwS'
                                                'luhYoMWwKhDkviCQPRYEoim7BgS5vg9kgQWv6iDvVR3lz5L9dZ'
                                                'bUoxn1cGxGPc1N0rPANd4NsM67gX4zyj5plP6oNoCb2kBDbsl+'
                                                'cEv22a5mENSnUs/2lfL7P5J5/yMzkeQXWPKF1L8vzNRL9n072f'
                                                'erJb+gRZJfYPBmuT+b5XsoHMQCIwu2gaC4CyNYoL3t7yBo0SJf'
                                                'FtLvroAgx9VaFnjyz2IQ9PYFcPMLqf9XyuHElXL6eZ2Ok9bpUL'
                                                'uwCvwXVpFNZCALTH1UA4cf1dDYNvL+20g9tn4MidaPafccPSfO'
                                                '0WNuRD3kRUj93yv1f68eR0MDCHIkPQtMudgABy42kHWx9Pdi6e'
                                                'c2ZjhlY6bjkj9N8ieONUPCWNln6iVvvdz/hWZwWmimYMkvMGO1'
                                                '1IvV0v8kf5zkL5b8NyV/WsxgFpjeawf80WsHHfLw5BQPTwxPSI'
                                                'e5CelkecqXBeoGXgVBq45oOfqIFh0n3wZBn1sjh1vLvDOgAgQF'
                                                '7JT6v1OHmkNV8C6lirK+DWTjt4G4pn0trG4v+05H2Xc6yvMHPY'
                                                'abgx7TmC/0LDBpZT0Iar1f+v9+uZ9jGqBwTANtZ/nzsszzVxtA'
                                                '0L+Sv1HyN9ua4bWtzLsPZd59KO9nnPT3cWb6WvIvlvwREWZYEG'
                                                'GmzZJfoOE7uR/fmek3yX+y5fy3yLy0Rer/HzNByJwbR2kZcVD2'
                                                '6wAU7JxnAKc8A5nbDsYnbQdzycbtIMjeRUHBse53QZAze5Aje4'
                                                'CHzSESoCuZ1IIuVidQdnUCBDc2waTGJjLEvdMK47LLTig4suo7'
                                                'EFS3IAIEndiwmdI2bIaoAA0uD9CwTp2keKiT1JLtzig4etdGEJ'
                                                'Q4xBt3D/Fmr6Lz4FN0niYWtUbB1iU7FKH6+g1CwZZ/x0KHv2Op'
                                                'qN4bBZ+MVOFEpEou77Q45J2WI3+6A4IqCgej4Mz2O+F8+50Ucs'
                                                'gHpx3y4dC/GQTVBSsoOPbSXYi7dJcqwnQo+GT9QxCU9+53yn33'
                                                'O1T3sSZTH2tICR1FApw+kN3pgwNgmNVNK4x/v62jxrd1EDErqA'
                                                'UZ3+xpASn/nm1Bmds1LaDs/YsWlN3xjtbU8Y4xZm03FJw9rhfk'
                                                'jOtFOZOd8PJkJ9a8i4bmt9GU1te1BdwfmEolA1PBcORPbfiRP4'
                                                '2v51RR85wqsBw/rQVZXPuhBTQONbYg61mPVaEU3X8Nhfdfk+Hz'
                                                '7krK593Votvd8frt7uzi1B+GOfWnaL8BKNglex0Iyts7HAVP7H'
                                                'kIgnoeIsfXNopQg/9XTgKSus5tQcX7tlLRvq1QNPFCC8q8eUNN'
                                                'v3lDsT7wDgSF9wlQgvsEqNXOPfCRcw/OvOEC6TdcKNXeGY/aO7'
                                                'PZfQM0um+g0JFuKNjS5ggIinnphZteenGU03kQFP7HEhA0/suN'
                                                'NPHLjeAScqUFxX93XhXKC+cPULDJZ67S6DNX1XzVEwVnhnqAMd'
                                                'SD1rxyxrWvnLno7xgo/juGEq+64Z6rbmy2OAr1FkfJPMMbn8yQ'
                                                '+/NdFqR/l0WZOX54LseP41vlg6C6wHwQlD3jiCqU8H9a4ef/tO'
                                                'Lg0BglNDRGDU6zRcGx50ZBzLlR5Fw0EAVb798CglI/dccjn7qz'
                                                '9vExEBSb5Y1bs7y5Ot8IglLb+OORNv5c/TgfHj7OJ2OdFtU6rc'
                                                'xkd+DmuDu06JQFCjZ8+JMS/eFPaodyOxRs+Bsg+m+ggpRBeC1l'
                                                'ED/33wZP/bfR4Sp3FDzxQhqMv5BGi3v5oOCJ3QnGdyfKHu2PF0'
                                                'b7s1V+AQgK6q7guO4KW1TfAU31HQprApzVBJyRXw5n88tpbvve'
                                                'KLjx7UcgKPmrwbjvq8GcnRsHgk5+5YG/feXBITtkntpxgk5G+a'
                                                'Bg+ykMvacw2W/wR8HaY9dAUJpWQcFWq++CoFAbxE9tkL08K8DH'
                                                's4KSpupQ8OLchyAo08MF//Rw4dwxP0DBmB/o4T8eKDjl45MgKO'
                                                'yaD8685sNxsdkgKFv1R8Hx3xWCINN8BQVbWZVAZ6sS2uaOKNi8'
                                                'pwIa91RQ2wU6FBxtVQWrraoo0TcQBafZ1sAx2xoqjCmmophiiP'
                                                '9yiWr4cony/NPTJCCyTycSYFq8lQSULJ1LouX/qyMB9quSSYBm'
                                                'fKYqlIiCKyRA45cRYPDLyPKfW0/aufUQHlGoCLVbh4tk0+EipN'
                                                'h/1ILqNv/UAsL9d7ag5rhQ0uwIBZuD/2tBiZtbU9Lm1uChlICb'
                                                'UkIpX49XhRLe5gUIyv7tgtb02wVjQXI7vJbcjrM/P6cI9fXAbi'
                                                'hY97EtaD+2pfTK3ni6sjcH9Z4Ao3tPoGAbJ5xsI/UndhU0b1lF'
                                                'dqmnWkB03+4kwGbBZhLgfCiCXA5JbbfyIgE3tx2kom0HIfqbE6'
                                                'rhmxOK1YW/SIAhebxWlzzeGJdUS/FJtdDoWqYI9VQnpvRODI37'
                                                'JrSgGZGJFBqZCEmeu1rIu5lCurlTIFGf2YJind+o8c5vlMgt90'
                                                'FQiqOXKpSSppcgSDO9XmuaVm9MWmKJPy2x5JQl2Ur6kmx1ZXE3'
                                                'FJz5YR/I+LAPFS7tg4I1myaBxaZJdHObEwouWroaBF2c7YKCPc'
                                                'p+AEGFn22kos82gvHGEsq8sQTmfzOcwr8ZDp77D5PH/sNQEpes'
                                                'muKSFcMfBSQgO3a/VhgT8qopMa8arKseK3ZVj9WEzkaK72yE4G'
                                                '5TYGK3KdQUuqsF2AxMbEFGv4mU7TcRjA/Ot6Dgsf+oQgnNLwNB'
                                                '8c/7qobnfZXgmlcgSBPaXjF92l7NUNrjGaU9F0X8pZRE/KUmG7'
                                                'rjPkN3rnPvB4L+ft8HBTvWfAKCTrx3QsGmKWtB0I/kgoLjuu+C'
                                                'hO676PXy4Si4ODEFBFnlDSLrvEFgd+ooCcg+vFMVyuGU65SSch'
                                                '2ys25rhTGsqZLCmyohOP6lItQ7Vn9SidWfEL0+tAXV63ZSnW4n'
                                                '5HT9qQXldB9Dud3HQNxKtQUVrahSTSuqlDrLByDI9LidKpSSa2'
                                                '/gzrU3ZJjVV4mc1Vd17tABB3bowI6Tbiouk26qka42+I2rDZu6'
                                                'DoCyrgPILs4ee8XZc5oyDQSVLRuA5csGcGoPAxzrYSCrvkOwU9'
                                                '8h7Dg2AQSFXhiOgp//eAhe/HiIZu8cgYLTIk7BsYhTZDptUIUy'
                                                'fftNCt1+E3RP2ypCtXB6QBqnB9CU1KoFHbY8SymWZyGoajaMrp'
                                                'pN412200SX7ZDU9HMLim/EFtDcNrsFxaffVVPS7yqOukpw0lWS'
                                                'qVW1UdO6Whv1+1sQFB4wQtEFjFCf3uqAgtMdSpUUh1L19Q0bFJ'
                                                'z2eBAICunZF6f27Mtp20NBUFT1AFxeLTv/2u8hYe33NGz1EBy+'
                                                'eghXr9wNNSt3U3InN9zfyY2L9hwGQUUFI/B6wQiOVX6Hzcrv5F'
                                                'vhhX4VXpyZ9ycIctQMUIR6bGwZpY0tg8x57cE4rz1ZtzpDVq3O'
                                                'QLTyRQsK6hLbAgz3klvQ/CI/iijyA6tdF1tQ48t8VSjWS6tAkG'
                                                'llplGzKlPrtf09CAr/d7QS/O9o1XlfRxywryOH369SQu9XqaNX'
                                                '9cAPV/XgoHOuMPqcK6Ue6ItHDvRl36KZoCuaSctCnFGwb+UG0F'
                                                'ZuoObbQ/DN7SFsPpwIfx9OpMJQNxQctesICArv4Imfd/Dk1C7p'
                                                'ICjV2huPWHtzyrfnQVDNtlEouLn/JWjqf4kcXazB2cWaxrz6nU'
                                                'a/+h3qdnwF1Tu+onf/xZDmRQxEZhyEZRkHKfbMyBaQaXO5BQVP'
                                                'zFaFkrHvEQhKGbHDaBixQ5uQrkHBjutDFaEaP7dCwbrvzUrQ92'
                                                'Z1Uu+eKDh3tRvkrHYj64EO2GWgA0d0/BwElV10xvKLznxSFwPp'
                                                'uhgyeLjieg9XDsrfA+Pz95D5sBs2HHZj09ZUqN6aStZjPbHLWE'
                                                '+2fpAOVg/SKVTvjZ/qvVk7KAsEpZwZhQfPjOKJpy+BII8MPxyR'
                                                '4cfZV/PAeDWPgouXgqAfy76npLLvIXzrYZi99TB57R1OHnuHQ/'
                                                'Ouqy3I8VSGan0qQ7HJrgVBmgWjjUKbmvQBCk6p/koRak7/Tni5'
                                                'fycO7/6fIlTnrJ44MKsn+3p5gpeXJ6X84oAHf3FgX2Uu6JS5lO'
                                                'k+EAU/37dJ+sQmytzqiue2uvLzf36EF//8SM0NbvimwY1d1h8F'
                                                '1/VHybTBEys3eHL2r6fh4q+nKWS5N05b7s3xRVmQUJRFuaWjUL'
                                                'ARcuA85JDjbT/sd9uPm7T58FqbT+nLA/D35QFsaXkDBGlDj4Kg'
                                                'jJjBlB4zGMos81pQet9fVKFoKx6DIMPT8izN0/KAkrWt8O7aVh'
                                                'zef70S2X+9OrqyE35Y2Ynj498qQi34zBYFJ1V5g6CQYf1w2rB+'
                                                'XPTFFyDIat9AFGz1bDMIOlnpioI1XZPgnXUSabzc8QMvdw5eeg'
                                                'ymLD1G91VPFFy84g+4ueIPOnzEGwWnRRvhRLSRwlr74qzWvjw6'
                                                'LwcEJf7nh3v+8+PU9Hw4nJ5PITsDUHDBwhuQt/AG1ZVrUXCS0x'
                                                '3Y63SHQr4tAEHh+/arQnF+0QCCNK+Lzmuai/w9wluj4OzeW5Xc'
                                                '3ltV7YHOKDi2uRUImvjCFie8sGXNVj8QpP2tHwr22BQBXpsiqM'
                                                'JiEAq2HhMLncfE0nzfoSg4yu1nEGTxnTsKjgg/DoKaX3ri25ee'
                                                'rFPOAChnSHfLGwNveXOcg/QRB5Xmu/riPFdffj71MghK6uGPP/'
                                                'XwZ6OT9GanAkpKC0DBuis3QNCaZ1oUnB15B3Ii79CM1oCC4ybe'
                                                'h50T71N1u79BkCY5IkBkZaAF/g8tODw+QQmNT1DPh3ZBwRGD2s'
                                                'H8Qe0oNd4Oj8bbcaSzAkLO1xEFR+77Chbv+4o8vh6EI74exM2J'
                                                'W+FdosyOO4bigx1DOSxoL8wI2ksWF9yx9QV3XjwxDSImptHEES'
                                                'Px4xEj2a5VBti0yiDPVj4oWHtRBbioUlmwLwouKr0MhaWXydnb'
                                                'Hwd6+3PIbjm33QW07mIACo4acBMEXbRUUHCJegcE3bEBFDw67j'
                                                '4Exd0n3b+AgtNOl8Ox0+W02L4NCg7/d58i1Gxra2Rray75qCMI'
                                                'qnPphYItz+pAUNovjvjLL46clLYEktOWkFXxIOxYPIjrHmyDmg'
                                                'fbyPnhUBzwcCgbZ+8DQa3aeqDgCO9fYJH3L1Tw5Ui89uVIDsvJ'
                                                'AEGrPXxQcO48grx5RAUrfVFw2ZwrIEgX4o+BIf4c0eoaLGh1jS'
                                                'zLArB9WQDPX38TBBn7Kqj2Vbiu/V0Q1HMQoGCX/PswKP8+LXkP'
                                                'KDj5UTnsfVRO8Aei4OYVJhA0/Yo1CtbM7AJNM7pQN+6FgrOV0S'
                                                'BIO7A/Cs4+uRRyTi6lSL/BuNRvMOcOioMrg+LI13sYCk5bmgyC'
                                                'LMd7oODFfU7Akj4nyHBoJK4/NJLtY/4HvWP+R/azfbDPbB/2as'
                                                '3g0ZrJI9kXhcxvV2R+u0J/L/NHwc3fXIN331wjn/8CULDd/Zsg'
                                                'aLGngl97KmwOuQuCeo6SvKMkb/tSGNS+lBZ3QvyqE3KJXQUI2Z'
                                                'cQL6jIISkmmJJioppJOhTseO4hOJx7SMOm9kbBKVljITVrLNkd'
                                                '7I+2B/uzJi0KBC3aPxgFG7/cAYLstwxDwfbf7wdBNjs8sMcOD1'
                                                '7z7gQIWlAyEgWHBp6F6YFnyWa7Dwo2pDCsS2GyvuCLXS74cnLE'
                                                'VTgQcZXcdvmjYPuya9C77BpZddZix85aThx5CxJG3qKiMQreGC'
                                                'N5D0jeA3fp/lhAwSFjSmH6mFLy6oMoeP6EChDU+y9EwTZ5JuiW'
                                                'ZyJdqA4xVMfJ9x7C3nsPybFQh4JH76iGMTuqSWvrhIqtE0fsXQ'
                                                'WL966ipveDUXB22k6gtJ1UcncY3r07jJ13HADXHQfI47oHul+X'
                                                '/bTiVzhc8Stt7uyFgu2az4KgxedkHz/nw5kB2SBo0SNfFDz/8V'
                                                'WY+/gqvTvlj4JzxhSCoM2DtCi4bPstqNh+i/6eoaBgc63krb1L'
                                                '80MB54UCp28ohVMbSqnIBVFw0NoKGLO2giYWIwoe/cwEgqLm6H'
                                                'DFHB2nvHkIh988pLO3dSg49JdqmPFLNa3xDkTBHlY14GZVQ45z'
                                                'XLD/HBc2Vf0AFVU/0JLBw1FwyZ6DcH/PQQq1HoGfWY/guqyTUJ'
                                                '11kqo/9MKqD714ccY5WJJxjpyrfFCwqTgbBOna+yG29+PFEbkg'
                                                'aEGBPwq2SC8EQWmgxV9Ay88f3oKnD29RzhIFLy1RuNG9BJ66l1'
                                                'B6BKBgXyoFHyolXx9EwZGnKmDZqQqa/gBRcHWfShAUt0iHgof1'
                                                'rQK3vlXkVa7DkeU6tr4su9nlakoLCETBNgNqwHpADRm2BuL6rY'
                                                'FsOboWBAVUalhbqUHDzjBFqMOc+7LAWFMoCMo43tACNB/1ztKM'
                                                '7h3weUJPDk/oiTr9CNDqR5DG93kL0IxL1jqOSzb6HezIvgc7Yq'
                                                'RltRJuWa1aOkntdOqFmXE6EPQ4YgDXRQzAIPN6GG1eT2U/1bUA'
                                                'wzIno2a5k7aLTTu2tmmHHmtPK0Ldd6wHJx/rgZmFQ8FYOJQ+u+'
                                                'zIoZcdMXraNxA17Rtqdi9rAY21A1RN3QBl5ohWHDaiFWafiVbS'
                                                'z0SrR6u6cmpVV4wu6AFrCnrQ5cd9OedxXww6MAsmHphFMc2FLc'
                                                'A6fZcqlKZHr0hAeEh7RajbiztxXHEntNa8VYT6/EZvFlh3fjwI'
                                                'Wqd5QgbNE9Akfe8vzpt8NCwwe+YkpWjmJHVgoSU7F1pi0a4cRa'
                                                'gTwqx5Ypg1+jZZgbbJiu6/6skCM9Nktk4bSaVsz2Vsjx6rp4Og'
                                                'Q1f6c8qV/miXsgJsUlZQ24GDWKDvb7Hg/1ssvesgs2jHWtDcDT'
                                                'cKbY7TGxKgC+mpCDU4ui0L1BSeUJqunVC/ndyZoyd3RkPABxAT'
                                                '8AE9c+vBz916YJ3VEDBbDaEc9z4ssOTZRDA9m0g7FzmyQE2/r8'
                                                'Gi39eU+sKZBeY8igFBs6tMFF5lAt2IBqNmRIN2dNp/JCDb4bxW'
                                                'GFdYWXCUlQXq5scrvvPj1SQ3KxZoml6vCNViSncWmNvPAQr6Od'
                                                'DRH3px6g+90NrxQ7By/JA+1fTjUE0/Oe/5ct7zqSJrAAusbjaA'
                                                'IMet91uAo7WXKpRFb/+hiLf/gGZyB61hUgdjp20fsNW2D9D06y'
                                                'JFqN1etWeB8T/cUJJ+uKFGz+3KQu57N4gzdaN/Kmy5scIWDW99'
                                                'QdC3W/py9Ja+WBc4EwQFRDmxNsoJDVXfQmzVt1T25lYLaPxhuW'
                                                'r6YbmypH89RfavB8PZ3CzN2dwAz8q35FH5FkzjRirZ40aqM0+1'
                                                '47BT7dDQP1Pm50yVPu/CAmMntQNB3fr2ZIGO291AUMT7Piyw7u'
                                                'EnYH74Cfn2688CHbcsBfstSyk44FoLMFnsVYWS2r2aBIRf+9Vo'
                                                'uParFtY3kW59E2RP+k+rmfyfsfrjNiyw0eqAYrI6oCrjOrF2XC'
                                                'dMv/lCEeq342w4epwN1l0dAIJ+Xtibkxb2RsdRQSDoU6Och7Ef'
                                                'WlZFgFVVBIWPvdICHM+dVz3OnVfW3ConAYYb3VWhhD55SgI0e7'
                                                '7XOu753jjwbit2vtsKraduVCynblQXOnTkCIeOGD6nQhFqwrxu'
                                                'LLDxDzto+sOO1rIdr2E79FUAvBQgmwkOLDD7v89BkI0jt4DIO6'
                                                'WqUNaE3yUBHvlTVF3+FCXD9QkJ0ISc89dMP3d+7zgNC8yun6wU'
                                                '1U9Wj1VYskDNuMtKU9Bl9fIia85ZZI3xEzqBoA1tbVlg07OR0P'
                                                'xsJFXl2XN1nj2mqdPhpDqdNoyV3WjsYzBcamsU2pPrnpGQ+7dD'
                                                '3vcOY3K/tyRAM36oItTVtq1ZoEf7zYpL+81qY2g7FpiyPENJWp'
                                                '6htvu0I1t+2hE15SZFqGzThbNtumDEoTaw+FAbGrGnG3vs6YbR'
                                                't3qBIENpDxaoazccBF2vt+Oiejt0DESwD0QKuNKHtVf6YMrmYE'
                                                'jdHEzXIh24INIBIyvCIaoinFY/cWSBjsGRIKg3DmCBuVXroKBq'
                                                'HS2IGMgCLT7ZDIKavhrMAs1/xYGg4A9rWoCmu8EotI1HG0mAZu'
                                                'pArWHKQKPn0tfksfQ1ZLfvogjVWPwBCzQ9iVSKnkSqj4a05eoh'
                                                'bVG37Ljiu+y4Wurfgcv8O6D1kmJFqHs6dObEDp0xZcF7JW3Be3'
                                                'Vgcld2Tu6KOjcbEPTbUxs++dQGLU8MBosTg2lKVzsWGO3qD2tc'
                                                '/cnzTW8WWJcxAQTNPtOXBaZMDoPUyWF0XevIRVqpb60Wg6AFe5'
                                                '1YoKZ4NTTfWk2PCp25utAZLRbGQNuFMfRv2SBuLBuE8c3bYGfz'
                                                'Nkr0rGoBpqmq0TBV1T7PfkICNOEBAZrZAVmmsy9JQPbzaq0wKn'
                                                'HSb+Pk/q2apgg15D8LFhg5dq8SMXavurBXe47o1R7Db19VQm9f'
                                                'Vee8seLwN1aYHv5MSQl/pk45YM0Cw6M6g6CQNjYsMMLXCRb7Ot'
                                                'GOkbYcN9IW6957gaABw3qz87DemH19DOReH0POT+1ZoObUpyBo'
                                                'T2w/Tozth+mHFoKg2U/7s8D0P1bCyT9WUpCnMwtMi9oAglpNGM'
                                                'QCPQZvBbfBW8m3h6kFmBzfGDX932iTr9eTAMP+XVma/bsCYhuf'
                                                'kwCN369aYXz43TsyffcOTOl+ilCHZrXmYVmtUfcsTvF9Fqf+/L'
                                                'QdJz1th5ruWYpQe9R2ZJta+R5G1CpCTd3XhQVmH7UE41FLmtu9'
                                                'OwsM/cMewv6wp2Wf9GSBQRfcQdCr0F7cFNoLLe8GgqBuw+1ZYP'
                                                'irKRD2agpRkQMLtHSfB4L+HdOfG8dI/vXLQND6hAFsSJDv4VsD'
                                                'CLI/O5AFpo/bAqfGbSGPhrIWYG20VzVGe+XOjVoquVELun2BRs'
                                                'O+QK2Nw1MSoLk8RyuM2269odhbbyDl576K4ee+ath3Mq981wo1'
                                                '5nWKo3mdOvtKWxbYOOiUUjfolJp0rQMLjL97X4m9e19tn9iZLR'
                                                'M7Y+jFViDokn03zrHvhrm2tpBta0tPFvVg86IeaFjiCmuWuNLx'
                                                '7+047Xs7TMrTQnKelrrN78MCY0dMAkGjejiwbw8HLPplNgjqlu'
                                                'TIAkv8l0Cx/xJa/Y8TC0w3rwFBHg4DWeDzeZtAUGTGvRaQ7apV'
                                                'hRJ3tZoEOF7eb9Rd3q9dgNKPUfpx/t0ATcHdrBn9XlFov1eQ7f'
                                                'mBItRi7w9YoOngPEWoRxPbcGpiGwzvekiJ6HpILT/VnstOtcfg'
                                                'kEIlKKRQ/WZLJ47c0gkdE18pLomv1IX9u3JE/65oiOgKgh6vt+'
                                                'G69TYY2m4gCOp1zJbtjtli8IlRMPHEKOr5U28WGDF9HMyfPo6e'
                                                'TuvLAku6zoCyrjPo5KN+LNBU9SVUV31JLuOcWGB4cTQIctvozA'
                                                'LrVmyEmhUbqT7yDtVF3gFd6UxVKH+kV1J6eiWEjy816saXaheH'
                                                'mEmAJtLzvPD3+vwFCTC8ztUKY+fb78n69nswtApShDp7lgULjL'
                                                'y5W4m+uVu9uc2SBYYfv6CEHr+gXlphxTkrrLDx2ydK07dPVG1/'
                                                'axYYn9oRBE2P784CS7b3g7Lt/aT+95T63xM1Kz3BYqUndbzSi6'
                                                '2u9MLshaMhZ+FoKkuQeTNB6sHCaWCxcBqdce/HGe5SD1Z/ASdX'
                                                'f0FrZRdeIzux4UQUxJyIog6PBrDAxeu/B0E/Ot6iJMdbEJ//nZ'
                                                'qS/53isaOCBIT/ZqUafrNSOs963PJ39aCb1dqomdVa+y7hGWl2'
                                                'PwPds51ax2c7jdmeb0lAuM0wRajfOrbmaMfWGLxosyLUeXPa8f'
                                                'w57TC9KEPJLMpQn83uyM9nSz3wqVQavStVV/suLLAkvw0I6p/c'
                                                'jR2Tu2HRm15Q8KYXNVf2YIFJw4dD8vDh9KDRjk2Ndmj4AmHNF0'
                                                'iP8vtwdX4ftPslGGx+CaYeK2W+WCn14H04CFrw1JEFmpZEQvWS'
                                                'SEr/cAALLNKsB0GRl663gMbmHapQXoeVUlNYKZheDVWF4vNZDf'
                                                'l+VgOGc2uMmsw12uTzjSRAk+qiFcb0mNckwHCnixJ+p4taXf4B'
                                                'Cwz/Y6kS+cdS1eDZlgUaio4r0UXH1Sf6DmzWd0BdbrHim1usOn'
                                                'frzAJTjO8Vofoe6coCfT+1Aa9PbWjCKxue+MoGG4sGw/OiwbTe'
                                                '1o4NtnYYMdkfBL3U9OEmTR803Z0AgjizL2dn9sXsyDAQdDPQkQ'
                                                'Xq+i8GQd0POrHNQSeM/nc1CIr8rKAF6DwPqB6eBxRLxxISkB07'
                                                'QRVKcVAVCTAkZRmFtun2ExKguRIUILJO5b2k9LyXoEuo0Qrjrj'
                                                '0ajt8j/RGmK0K1fWfBdu8ssOjYXqXk2F71YP/2nNK/PaZ75CpC'
                                                '7dm6EwssOvdMMZ17pm4+as0C6/Z3hur9namrlQ1bW8k8+7kTCL'
                                                'rvZ8sCIwZ6g6DWnr3ZwrM3Bj0bA6OfjaFDL+055aU9Jl3/FASd'
                                                '2SHfww7ZPy4shKgLC2nAy/7s/LI/phWtBEFJD3NbQHzMKVUoE+'
                                                '8Uk4Dw0K9UoRQPNZEAzdxXRqF9XV1PTdXSH0u2ZWnubQtIt/iP'
                                                'BGhKT2odS08ay3a+IwGa1f6KUO1yWrNA09QdilBfvG7HAiO/yV'
                                                'Kiv8lSzX93ZIHhW2sVob493IWbD3dB03VLqL5uSbre3VlgxH17'
                                                'WHzfnso/7clln/bE0Fp3CKuV/jhb+uPsXuj4JhAEnfC0Z4GGPl'
                                                'Mhps9UennbgZtuO2Do5HkQMnkeXZ/Qn4sm9EePA8tAUPXsyy2g'
                                                '6SmpmmekGNffIAHxZzapQhnUqpxcWpVD/P0+asr9PsrO2lqKr5'
                                                'X+OAKMhhGgTfB8SvGeT8HQ9QutMJbUvCEBmi8cFKF+E9OKI2Na'
                                                'oeH39YpQqwtlXixsiynyroX64+0OLNB6RKki1E37OnPMvs5o+a'
                                                'QVWD1pRTkDpD8O6IYROluYr7OlVUt7cPTSHmi3xxXs97jS5S12'
                                                'nLPFDnX/aEHQ1S/7cO6XfdA3ZBJoQyZRXG8HFpiSNxtS82ZTh/'
                                                '2OLLDksyUgKKzwQgsIX3NdFUqofSEJMNzbowrlYv49ys6/B+Fj'
                                                'A1TD2AAlpbyaBGR3TDYKrS7kHxKgWVURYFhZkaUZ+YoEhGtaKQ'
                                                'ZNK7UrfsDW+AGmfDFfEerU/W04ZH8b1EUeUoT6wbn2rDnXHjW/'
                                                'FypC7fJDJ7b+oZPsN6+U3PpX6naXrhzn0hWzd3UFQT9utmGBSW'
                                                '4DId5tIE0/acsCY6+PAkF1yb1ZYMq34yD123E0b2Zfnj+zL/qO'
                                                'nAGCuL4fZ9f3w4jWi0BQqxFEmhEEKf1MqlDsTuaRAMu+x1RN32'
                                                'PK3tg7lBR7BzQvZqiOL2YojlcrSYDjjntG3Y572pQvzSRA4/jq'
                                                'T43HKz+P5S9IgGZvnlYYy2rfk4DwvCAlMi9I9VlgwQINnolKtG'
                                                'eiar/bkgV6tL6o+La+qG5da8Wxa2VevvFEEeoGF2sWGFHYEeYX'
                                                'diTnn7qzwKQz/UDQjwU9WWDuPk/I2edJr/N7scD47aNB0IIkex'
                                                'Zoip0Ggkxe/Vhg9t4vwLj3CzJtPt8CUkobVaFo3K+SAJe1Z1XH'
                                                'tWeVVN9bJCC+erWaUr1aaTpSQQIaCzqqmmsdlaRlj0mA5u/3WZ'
                                                'p/3gcEH39GAnQ/xWvDf4o31nz0luo+egvZFcOU9Arpj0OkPw5p'
                                                'jYb8zUps/mb19pftuPhL6Y/D/6dkD/+fWrGgIwts3Fmp1O2sVP'
                                                'cM6MKJA7pg0Is2IMj6SDcW6OLcGwRF18n3UNcDG6cOB0FPXtix'
                                                '+YUdlmxFELT9Rh+Ou9EHo/ODYU1+MM38zkHmVQdM6jcHBFkUnW'
                                                '0BcXc1FH9XA9ZHL5GAkqYralHTFaWk9DoJSIndoQrFIaqUHKNK'
                                                'wWA7VBWK/Tc1JMAw7luj0AZdbyQBhudDtZr/hhqdkl6TY9JrMB'
                                                '22VhoPW6tWtR+wQM2qZYpQewW0ZbuAtpjulqYI9d34DixQ53Rb'
                                                'CXK6rf7RqzMLDG6vAUHjT3RlgY4GGxD06L0NV7+3waKXg6Hg5W'
                                                'D6ta8dn+xrh8FR/iDofNs+LLDk9QQofj2BHKkvCyzaFQYFu8Jo'
                                                '+VwHjprrgCWbwqFsUziF9tCxwLKHlSCo6ZEVCg5e8lTRLXmqJn'
                                                'tVk4CUgF+M2QG/aJd2H8iR3QdiWuAmEPT0kD8LtLErBEFhP+tY'
                                                'oGNeFTjkVdHLcXpuGqfHNOt6OGFdT5bPdC0gY93PJKC9b0+29J'
                                                'X77OoOBa7uFLLdhwWGHWKYfYjpiyjk+VGIaQkPIDXhAZH1ByhY'
                                                'tzlcCd4crmqWeQSILIppoOyYBkj/7P8t3X1cznf///GGWZzNkh'
                                                'DLhFhIQhdHdRyv19Gx0JxmbWNrFmsWi7lo5kusWSfNGs1pO0Oj'
                                                'bbHmMs5mIXS8Xl1fqySEEJIuDjSLs5ltv6fb7ffH/Q+3eHd79H'
                                                'l/3hc3lJ0CpwW/ZgLrnGhvjYj2Zpd1B2jwugPicdiPXzzsp8bn'
                                                'cwmk57EBDFrTfyKV9p8oYY5TCCS63/YnKCc3X4C8n6+wuj1fYS'
                                                'q9/6IC91i7mXqu3SwbT/oosP2rJwjk3odB2vFhEHvaagjE7jdS'
                                                'YMfsK9Qn+4pMqDWrd62Zp6c2EUj6mWAFDgtroelhLTJptkV9Zl'
                                                'vY/Eobgbzxo0Vn/mjhqLntBNKF9QQ4NzDZlB2YbL1za5jabg3j'
                                                'xKiVlBS1Us79e4LW/XsCpy85TLuXHJZN+QGalB/AdY/KqfZRuY'
                                                'x0Ipy/iGPWXaIV6y7J97PNChxx4wbNvXFDqr4OVuCOebfp7rzb'
                                                '0sdgUeCG06106XSr2KUtfYLmLl8vkcvXU1vtHWmpxfmobkqQ3b'
                                                'kpp07GD1Zgl+pQAik5NE5LD43j9Fm7aPesXfLghL92nvDn5JIC'
                                                'Akn526jAaannaXfqeWkoZQX2Lmskr7JG6R8SrM4h+PpYmgkk7Q'
                                                'UPAons+9MTZEvuKx3Jfclx1n8FKPbRaQEKXbPNGrZmm+mtz/pq'
                                                '+Gd9OfGLfpT0RT951OahwD3Svyb79K9ljtFXI4y+HNf3FK3pe0'
                                                'pGZgape2YQR18/QyDPD2R1HcgcYbxKs41XJfS7TgKx+yTdCDn1'
                                                '+i8Cmd5nuYT1WU5po01WME3ZcVFCd1yk3v33i33//eQ5Z6h4zB'
                                                'lKI77tqe7f9mRvvwwTWPP6j9Dc/iM45uqnBPKH10R97DWRc5Iy'
                                                'SZIy5c/+gfq4fyDWv0qsf5XyeNkI/mPZCE1wiCeQWPMzvMr8jN'
                                                'ot+MVkv+AXa4fpIIGE/jD4CbIb8oYJrL3DL4hD+AVyzVgrbhlr'
                                                'qa51pdS3rqQ+fX4Txz6/Ua4hwQg5B397XjN+e54jd75GIPdivL'
                                                'Qjxoun3/iRQFrm+TLoiu2nCCTDZzQf8hmtbsnf0PDkb2RwbycG'
                                                'jT04gEASGqoJJObzr6xg2nJlACVdGSDe9zOeoPim4QJ0f9Ve6V'
                                                'i1l3Jc6wRoyxvLrWlvLDe9aOmnHpZ+bCgYQiCejiYe62jSrPrz'
                                                'dKz+vCy66M+gUX6FBNLUOo5vto7TqBd207wXdkvH/cH86/3BWr'
                                                '/unwQSu/Qegdhdag6Kv9R8KiXxYwIx9vjXExT/wzormGa710q4'
                                                'ey3lumUKUGTNcwK0cOpojZ46mo1nviE68400v+GtwDXGA1RnPC'
                                                'BHF/tq1mJfTjpxijaeOCV2QwIUOPZwKa05XCpflQVpUlkQ1xtr'
                                                '6ZKxVgpPmhTYHHKRLCEXZbM7K3DGkqu0b8lV8fE0K7Cb/w16wf'
                                                '+GbLGZFdj2ZxPd/bNJHFYHK3BHT7yvPW/LyWvBCpyd0EInE1qk'
                                                'eoxFa8ZYeEVqK4FY37NoznsWjl7YRlEL2+TXLy3a8aWFE/5uI5'
                                                'A5ey0asdfCcUvaae2SdmnJsSiwg7RTH2mXraXPa3Lp89yVHEad'
                                                'yWGyOGmUAkf9/iWBTJntpcCR236kedt+FKe/JylwbvExAjE6Gx'
                                                'S48ukiAvmkV5DG9Qriho3VVL+xWkaPN6nHeBO7vHyBBr98AZ+R'
                                                '1CbE0b83YPwGGXSD1eUGs/Oo6zRg1HVZfcysscfMXOjbRCBfvh'
                                                'asia8Fs3FFMwWuaJYL+cFanx/M5kEtRINa5L/9LZrZ38JZr7QS'
                                                'yNTXLRr6uoUzx7cRyK61Fk1bi/X3HNbfc23S/wcL7o8WTprRTo'
                                                'kz2uXZD59Rhw+fYe+goyawntw1UIHNW/wJ5O6GEQrsMSuexsyK'
                                                'l3ljPBW48JVUKn0lVWjRRAWOeetnWvrWzxJ8z0/N9/w4/O9cAn'
                                                'nON1AdfQPZ/j+VBNKQbFTgsMHnCGT3DFJgx3cvk9O7lyVkCyuw'
                                                '8+/XyPH3a5K/xqzAjm/fpD5v3xTbyGAFjqu9RWtqb8n/dgdr1+'
                                                '5gztx9m7J235bIv4IVuOFhC1162CKBwRYFduxoJRD35RYFTt/T'
                                                'RiDp888+QWaf1VYwzcJ5fybO++YPo0zeH0ZZi5/tp8D7vnYlkK'
                                                'Xj3RQ49/CHlHP4Q/lYRusKGc1ds5Lp8axk+WO/tz7e780xkQdp'
                                                'eeRBeXTAV4EN3+QQyKBpAeoyLQBf/zICCW0LUuDsI7V07EitbK'
                                                'k1KXAD1tpLWHNv+LICGw5dJb9DV+WCwaz1BjNnz79BJ+ffkPIH'
                                                'ZgV2c7tFrm63pPyzYAXuGnqbHg29LfuaghXYeXsLgSzysihw0/'
                                                '5Wat7fKhvmrJPEOeuoszpGOqpjqPSPPAGKOVBlBdPdbg+ko9sD'
                                                'ij+WYcw9lpGzsXsfBY4Z8LcpdsDf1tufumrLp67c0jyTbM0zsZ'
                                                '69qMDpuzbRvl2bZOcJL0094cX259Opx/l0sQvxUeCarONU++Tv'
                                                '76cYFLjjaBHdP1okh8YFaca4IM5wqKEDDjWy8mWTAifsu0Ab9l'
                                                '0Qx1pS4BV8hZbzFXm2g9Whgznm7eu0/O3rsj/XrPtyzRzzThMt'
                                                'fadJHoUHK3DDv5vp0r+bZer////zK8a1UOy4FgnLi7Oa8+JMj3'
                                                'ueFaCWHuEC5BK3WYAcZ2AfnoF9OL2vAOXObBag3EWf5sQv+tR4'
                                                'drW91q22Z8f2XJN9e661ZqGLAtcHGQnkFxd3zXJx5+gR62jBiH'
                                                'Xy6388teM/npxx8Ds6fPA72XJsogLXRRwhkNWj/DV2lD875+XR'
                                                'gLw8GTgnUF3mBHLHsNN0f9hpufPzk5+la+SY9ecIpPR9UuDQjM'
                                                'sEkvEDK3DWhEY6MqFRum0wK3DKupsE4u4VrMC2jlt0t+OW1AQ4'
                                                'Ul2Ao4T9+t8nyC7njRwwxs69LUApK0YLUM3H6QLU9P5OAUq1GS'
                                                'XNZqTZC89L+MLzFOYeYTW7R5iyPuimwI0hn5o6Qj61hrT1U+CO'
                                                'b4eR7dth8kONm6bV4P6+cxmBhE0ao8BJdVtpY91W+ewvb43/y5'
                                                'vNn2UQyPMtvura4svGD6wEErkyQIEb2svoanuZLO5tVODEUWcJ'
                                                '5F6rSTtaTdx57yLdv3dRWqewtkxhzmy/ShntV4Unm9U82cyJ/7'
                                                'lBCf+5IV330unRvXRx/8BDPD7wwD2HnxC/ITvEMGQH5Ub1M4F1'
                                                '49UrknT1CsXk3reCyeh6UoD2Td8gQLa6aOmoi6bQwCIBiv1Orf'
                                                'HfqWnfl/8TIPPk60a3yddzhjzso64PcR+634NA/ho/RIHtR79N'
                                                'DqPflsSvX1TgmS99RSATBo5X74HjOcpxD81z3CMHv/DRjC98uO'
                                                'n7bLrxfbaMWIX3Z5WBHd8vJqf3i8X/tSA1vBbEhqQa8kmqkWnv'
                                                'mxQ4tGc9gTQ0kQKHJlwhkJboO9QcfUfspnQPspvc/ZRxSjaB2D'
                                                'n/Ye3o94fJe84SnPWWyJG+CZLVN4HSPA6ZwJrlWSZAuW9vsYJp'
                                                'RUi1ADWZI6XFHEmBazeKYe2Tn0FxRICMLb3E3NKL6vu1CpBda5'
                                                '+c+NY+xhEJvdQ9oRfHxFaaYmMrrfO8Bilwx3kzgZzf667AKW3r'
                                                'aVvbevn6D0/d8ocnt3T/gdq6/yBrn5qkwA1zfqGrc36RIxH+Cp'
                                                'z4WT6BzE8I1KiEQI4+fJoWHT4tn1QZNa7KyIVt5yi37Zx8/7Ej'
                                                'g8asfZZWrH1WbAl/EEj8nMGmyDmDrWnvlBOIS8peq2PKXpPxqy'
                                                'QCaeke8QRFfvbMExJ+6ReZeekXsnv7SA4Yd+tNSdebFGM3UYCa'
                                                '16ZJy9o06tb/B7Hr/wNl9vKVfb18afm8ixIz7yJteZmt8S+z6c'
                                                '1N3RU48s5GU8ydjdb7+c4KHPr1KDJ/PUralg5TYPs1HxNInz1j'
                                                'FLjHK9vJ/pXtsuq1CRr72gSe/u0hAnnB3U+B0wxCIA27AhQ4fk'
                                                'M5rd1QLpWnh3HF6WFqf/r/CKQppD/fCumv5s2jCSQ6sAeD1rz5'
                                                'jQmsnp6XCcTtwjgrmLpm7KKHM3ZJotn7CUrz9Kd0T3/J7P/9E2'
                                                'SOvmqEnDWJ1yUu8Tp5ju0m3mO70dyAYxIZcIyyJ24UoMeXI6Xr'
                                                'ciR5Li0VoMw9mVYwxfR8JEBu+3ub7Pb3tv6n6TlNbnqOPUJ7k1'
                                                'tob2m/OkRtV4dw/DtzCGSTo4cmOXrwiqubCcTx4/EKnDZ5L6VO'
                                                '3itDyn3UtdyH7T4/QSAZAeMYsOf+QN/U/iBpf7ozqP3jBAKh9E'
                                                'EM6pb6ErmkviSdI3rzgxG9NWzpWVPo0rPWxSfbCMTu9ZRTdq+l'
                                                'BIVdzaIZV7PEzvD0E+QdNP8JWeCYKFGOiZRp2WnaZ9lppa5KMX'
                                                'ZVUtjU3VYwWeIqJCSugmoWRwtQ4WcbBMjD95gAZfh0k0yfbpRc'
                                                '2i5AdouePgVBvTb2VvuNvdn75gUTWD+9N0jj7g3izPcmU9Z7k2'
                                                'Wez0gFrjnwOYHMeG+cAqe8mUYghkwfDsj00eljT9CMsSck5NXx'
                                                'DOp4eg+BOF57kUE9aDONoc2Su2MI5+0YotHZ7xBI+hfP8Y9fPK'
                                                'cuk+zJdZK9ePT5nUDS1tqZwGrwKiGQ0vhj1pr4Y6bSjEQCCfvj'
                                                '/SeoccWfppYVf1rT9LgA2RU/ygFj7aBGqRnUSHafBQrQ4S9SJe'
                                                'OLVHLP2S1AiZ7jBKgzrEGAatKHWxvTh5smH+ihIQd6cOSA7Saw'
                                                'Nm7vr8DZq8ZSzqqx8k7v4RrRezhHv7yKFr+8SmxbAhjUPK2caF'
                                                'q55Dj4MajdbSuBuE+YwCMnTNDIsYcIJGL1GH5n9Rg1ntxG5pPb'
                                                'hCYMY9DG/31EIOuWOjOo9y53Akkf1J1Bt7htMCW6bbAmBNQTiN'
                                                'vX06xgsv34HYE8HmqQrqEGatzn9YT0dNgt9g67ye7BViPkbBt7'
                                                'S5LH3iIHW29xtPWmlNePCNC65zZL/HObKftWuABFba8QII+SXV'
                                                'bvkl2m8NWPBchuqbsJrF1Vjgqcm/QcFSY9JxEFRgaNzz9HIC7/'
                                                'F8gD/y9QM1edpsOrTkvyVH8GDfPNJ5DpTRP5laaJmvn7EQLxPu'
                                                'vJE856avan39OxT78Xpxh3BnUJWE8gpddduOy6i3b1ZurszTLT'
                                                'tRfPcu2ljQ9KTB0PSqxpT7UQSO7AwJz4gYHGxg9/JhD7ew5PUI'
                                                'v9HLLZz5F9fZKeoJiwL01gJakRo9SQm0eWFUzfHy6R1MMlFJ4Y'
                                                'I0Bd8eukM34dPU45IUCJzg+s8c4PTH0+vos14S7ZxW8LglOJV4'
                                                'i/uEK6YuEVAgl828SgudcvEEjES0E856UgjZ9dQyAOUQb+R5RB'
                                                'U4YW07ahxRK2xIdB972UTSAn73sxaP3Wn+j81p/E/d0XGTQuLY'
                                                'nWpiVJSpsrf9vmqjWfvUV1n70l32zvw6Dx57sRSNxXDwkk3lpp'
                                                'hBzvhwUEEhpfZA2LLzJ5FiQQiMuDD5+gLRfaTWC9lHZKGtJOkf'
                                                'nfw6xgClnSIKFLGijznkWA4janCJDn23sEKHfWSAFqybRwa6ZF'
                                                'fb5uJ5DcMb4MOr3uJE2rOyleI5lBO5ddJcB9Pc0E1syiCgG62m'
                                                'ph0LAFNgLZsM3CoIle7bTRq12yZ5j5xAyzrjh2g1YeuyG1jeMZ'
                                                'tPT4PgJxL7QwqPvFdhp+sV06Ngfzvc3Bagy/TSAZvxv44O8G9Z'
                                                'xQSl4TSiVhXCeB2MXsNDYu25kTbrAwaGdVK3VUtYrHUROPPmrS'
                                                'hvEXCaT0R1cGja+aRXFVs6Rj6S4rmAYcqBDnAxXk7G3m/t5mXT'
                                                'z5Bi2afEMa//Zi0Md5P9FfeT9J+N8LaebfC8Xx5OdP0HuPn9XI'
                                                'x8+yXdFfJrCm/4q75JPvCbXeRgvW2yT2toVX37bok++fBLKv0s'
                                                'KgXV3t9LirXc5/bmFQ7442mtjRJhvKgxm0zreFQKZcZAZNfvo6'
                                                'ffP0dcn5KZCtPwVq+sgq2j2ySjz/5cmgi5d8R4uWfCct7Xh+7R'
                                                'btWGKjziU2iai3MGihu42K3W2Sm27hvHSL5r7fTiARrhZ+x9Wi'
                                                'Wxa0EkjNETOfOYIzsUcTgXjeNfLYu1gP5pwnkHDrRAZ1XnGEQF'
                                                'quDWDQxD6TCKS50cKg2SE2Apl2ysKgj4+301/H28X2loXvvIXP'
                                                'H9JGIE1hwXwrLFgbPsJ97KNm6bmVGNQ7soG8IhvEIdCf/xHor6'
                                                'VP5xNI5bARDJo4Zi0ljVkrjdbLBJI52cOaO9nDlFVh4V8qLBr9'
                                                'oJ0WPWiXsPUWBi1tbqPC5japLwhm0C7XFnrk2iJxJcyg7oWN5F'
                                                'bYKPVY70Cj1pymBWtOi/NfYxnU/dRO8ji1U7KkB4OmdW43ZXZu'
                                                't4YdG00gG2alS8KsdLLttnD7bos2RrQTSNxAC3860KIxM1sJpO'
                                                'UnM9/+yazxXTcJZHOFkUHdG8+RR+M56f3uRAZ1nvAzgSRf68+g'
                                                'kUPHEUj0Z6m04LNU2bcl8AmqmTVUgKJc9ku0y36Kmmnh92fief'
                                                'q30Wb/NomfGsz/mhqsPV5tpp6vNkvnJ8S/fUKa3KOBvunRIO4P'
                                                '/XjEQz+NHZhHawbmSe4Hw1k/GK4dyasJpHJwPYHEn/qnFUx2dj'
                                                '1OQED6VzYBWuxdL0BbWqdZ41unmXpYgxl0S/cWAgk7yvzqUVbz'
                                                'vxqJ/tUoDSGBfCUkUGMqKmlpRaV0rRrLv68aq652OwjEPqI7gz'
                                                'beSTSBtWvAUAIZ4LtfnH33U/iOVAHyfitQgL681V8Tb/XnFMs4'
                                                'SrWMk6wUM/+SYtbMkpt0pOSmjNpnZFCHyHPUJ/KcJP01gb/6a4'
                                                'KG3v4vgaRddGbQlAEvUuqAF8WleTuBpL8RImlvhFBG52jJ7BxN'
                                                '+YZ0yTWkU2lxDwVOm55iAuuIpzzV/SlPDinaSeYizI/fLHz0N6'
                                                'xfm2w0apNN3LAeDMN6EPcvG639l006bBYGrfzIRlUf4ePN+Hiz'
                                                'RWeG2wjE6bKFQVu8bGTzsolrmYWfL7NoZns7gbQctDBo3Np2Wr'
                                                'u2XVbEWfjjOHy+020EMmW8hUFdD7WSy6FWyfg5mA/9HKxRF2/T'
                                                'vIu3xeGWmZ+9ZVbXG030wo0mGbydGdTTqZG8nBola4WJj64wae'
                                                'LkegJZ+VsAgzruryCQdWN9GPB+HqeJkcel97djGDRk+HayDN8u'
                                                'rvctDOrxuY3GfG6TxLsWBi1cZSOQ7BYLH2958j2ObQQSes3CL1'
                                                '+zaAbZCKSj2sK/Vls05GkbgfQ4buGnj1vUfl879dzXLus2Wxg0'
                                                '2bGdtjm2S/p0C/80HX9+cBvtG9wmcdeD+ZPrwWre1EKWTS3SNT'
                                                'GYQeu7NdP5bs0SPsDMoInN12lj83VxmUw8cDJpiPEygQyYFsSg'
                                                'qctr6PvlNRLznh8ve89P6/oone+jUmjx4gKLl6ZP/ZFA4mrcGN'
                                                'Ru5zICaXryvZHu4f2Ps9HVOJvcbbMwaOgiG01bZJOrN7A/3sDz'
                                                'mW4jkNJzFi45Z1GX520EEp9rYdC4CjzfCqyX31kY1GdaO02c1i'
                                                '5Z8zG/5mO/iG0jkJDeFgb1HNFKHiNaJfr/ghl08e/NBFK4zsxF'
                                                '68y6b+VNSl95U2IfEq9+SFpZeIVKC6/I60uMDLp4Ux0t2lQn+c'
                                                'EGBo3dU0Rr9hSJ+1ycd+dO0BWFh2hl4SFJ6hzJmzpHanbfRDrZ'
                                                'N1EqR/Xj06P6adqIIZQ6YojUYn6DhmN+z8b87riF+X7Loilv2Q'
                                                'gk9xL2n0sWzRlno5PjbJJSauHtpdgPWtupsLVdnDG/QSM+aafI'
                                                'T9rF5RMLD/zEoo6lbQTiOM7CoA3prXQ1vVWcDwdzv8PBald5m0'
                                                'A8r5l53DWzRlU2EciGr5hBH9+9Ro/uXhO7+SZ+ar5J4x3qaZ1D'
                                                'Pd7/ALz/AWr/SgX1fKVCaux8uNrOR3Mqj1Fu5TEJfX0Mv/z6GH'
                                                'UM3EYgjWGD+VrYYM0+NJVAjr1px6Bu3V4zeXd7zboZ8xs06j0b'
                                                'gXTD/AZNN9lojwnPF/MbNLW7jb7vbpMZxywMGvFTO4HM/srCoC'
                                                'EO7QSSP83CoCn92ii1X5ssuhrMoPWftRBI1zjM73HBOr3jFoF4'
                                                '9zEzaGzpdQI56U8MGv+Py5Twj8sy3CeIQetG1BCIj8mPJ5n81L'
                                                'BFKHCLSE5PLz7V00sr1+6mqrW7cWZ0Y1B77yXk4L1EjB692eTR'
                                                'W92+P2sCa8f0Kro7vUpK7ydbc+8nm9Ixv3/E/A75p43M/7TJAs'
                                                'xv0MxBNjoyyCZNauGbalFDWTv5lbXL3VS8H6mY71PaCcQlysKD'
                                                'ojC/l7eR1/I22fOMhUGjB7USiPtHwTzyo2BNaW2mba3NUvuJmU'
                                                'G7Im7Sw4ibMridGNRt5xV6YecV2TfbyKA9QusIxG6EgUGbAoqo'
                                                'MaBIaOwEBp0+8BBNG3hIKG0kg+7r/gWBPEp1YtBsu8EEkhvRQi'
                                                'DmvLE58XljjXZ/fkUgPRzexv3sbYrB/P4I8zvJ00abPW3ijPnd'
                                                'D/O77nY7gRgPWDjoAPpj2wkke7WFQW35bQRSN8bCoImprQQScy'
                                                'CYlx0I1ji5TSDmi2YGnJ+baPmxJqlbzwwaUXGN5lZcE7dZJh42'
                                                'C+fd2gt0tfaCxGsAgz6+V05/3SuXtHOTGLTQ/xiBZPcew6CJr2'
                                                '6lja9uFZfLg3jQ5UGa2RxCWc0hYuf3N/3l+7c0fhBsAmt42WEC'
                                                'aSp3kpZyJ6r/u9sTcv/SUem4dJT6YH6Ddj1lo4dP2ST5qIVBM3'
                                                'a304Hd7eKZZGHQpmdwPnqmXS6FWhjwPrYRSNjlYAadvqKFpq1o'
                                                'kRaPYG71CNakK7cI5O7TZgZ1PHCd+hy4LoYxxAFjSI2XLpH50i'
                                                'W5OySIQT3rqsm7rlq2uPjxv138NNRRaIqjSI/ycfx0+Tjt+GUX'
                                                '3f1ll3TZu/Hv9m4aqYsIxG5DL35qQy/1/qrSZPiq0uoeWUkgiU'
                                                '7fW+Odvjd5L3uVQBJTv3mCIhv+YQVT9O6rAjQF8xu0pbidQDp2'
                                                'WvjeTux/Fpy3Le0SO8/CoK6L22jw4jaxe9rCoEnPtdLG51qlbn'
                                                'Ewn1scrMkXmwmkdIWZy1Zgfltu0iPLTdlxjRi0x0dXCKR0qpHL'
                                                'phq1sXsd3eheJ03dDXyru0HdKwtpVGWh5P7Pm0FnRmcQyKNpIx'
                                                'n08f3PCWRfXyfe39dJwy4MoJkXBojLgWYCMfeOyYnvHWMM37mJ'
                                                'Zu/cJCsezsHeMYfMn/ibwLpy/CWJHX+JXMMni1v4ZMpetF2AIv'
                                                'dbGLTH/7UTyMxYC8+KxfpuxXnf2ibJHhb+jwf6k1sJpHBPMBfs'
                                                'CdaazNsEEnLWzC+dNWtIWhOB2K9h7rWGNWrvNZq395o4TTYxaH'
                                                'LKBdqWckEG/BTAoI+/LqdHX5fL/f2TGNSw6yj57Toq4fmj+a38'
                                                '0Zo5L5lAekYNYtD4QRZKGGSRxpN/Eohb1gSTY9YEa/j3GfTm9x'
                                                'myb9vAJ6hj1B8msDpYsgXIUNJh9S7pMEVsOilAxi8PCFCap6sA'
                                                '/bXRwqAx3dtpRXfs31Owf0+x6PSn2wjEsz6YQVOjW+j76BapHB'
                                                'HMoKnlt2hH+S1J/pN565+s0zddJ5AtrsT/diVdvO8SgcQ9E8Sg'
                                                'meur6cj6aln7my+Ddm62Ekha0jj+IWmc2t1Oo7+a0yTxx6EM2h'
                                                'UaTZ2h0eJ8w57737BXt2VFJpdlRdbpy8oJJDZorzUmaK+pceo0'
                                                'Aoly2voE5U4YbgXTkYgGyYxooNjXvxSgzr3zpGvvPMp/5pbkPn'
                                                'OL4vNSsB6mGGsjsf9H4v2f30YgM7tZGHTL0624v7ZKbHQwr44O'
                                                '1siyZppX1iwZS8wMmuVxk4553JSus8SgMVOu0NIpV+SIj5FB7f'
                                                'QsgSxu8uclTf7qM6eQ/OYUyoECbwZNOn6QNh8/KNLpzqBZ1xLo'
                                                'yLUEOZ/Ql0HD9/ajiL39JP63JgJpfHNHTu6bO4wefl8SyOPr88'
                                                'TuxjzKHTPMlDlmmHXDqgZJXNVApQnTBGjeg2SJfJBMoRvKBcij'
                                                'eo/Vu3qP6UCrvQJ7FxSZDAVF1mmjcN8dhf19Uyvt2NQqDj8GM+'
                                                'jj3bfpr923ZeJpM4PWfdlEIO4xzCNiWHusv0bd1l+Teh8Tg2bO'
                                                'uUBH5lyQiI0BPGdjgNomlhNI4aeTuOjTSerc6yg59Toqw1ePZt'
                                                'DYj/9Daz7+j+xud2FQxylMDlNYHIyPCcR8e5gp7PYwa/SyAwQS'
                                                'Md9VIue7UseUDlPXlA7r5r0nJWnvSWrZ84e1Y88fpsKR2VI6Mp'
                                                'u6jmcIkPPcgeI4dyAtqvxToiv/pNwlE001SyZav1w4SBMXDuJ9'
                                                'gRZKD7SI3blgBq3BPgmS9kIw73ohWGOO3yIQjwfMLz5gNSy+To'
                                                'GLr0tuH2LtQ+q48hI5rbwkpR2BDBo2sZpA7Kp8+akqXzU7WAkk'
                                                'YcY4Xj9jnHYNTCOQTuNQ/s04VHMr5+O8Nl9ijfYMWvOSmOpfEm'
                                                'v0mlLc40slekmmNXJJpqlr8GQCyfxiu2R8sZ3C1hmskesMpjhn'
                                                'vFvOlyj++CYBisuZI/E5c8hbmgXIfHNxTvzNxcbnXZzU1cWJsx'
                                                '8PIJA7M0aqbcZITv37c/r+78/FMSqY+0bhvn0C9+0TzRK3wMyf'
                                                'LDCrR5+bBOJXSgwa9fwVAlkwzMigUavPUuTqs5Ja6M+gbncKyP'
                                                'VOgTze7M1/bPZWW4+DBJL4nTt/8Z27FpatJ5CEh44M2hLjSCAt'
                                                'hpsEEnkuJyf+XI4x5vQGAomr+kDiqz4g87neJrAOtl4VV+tVqj'
                                                'z4qpQefJUSPsN+9tk3ZFtZKUCle76z1uz5zrT+q16a8FUvrmms'
                                                'NIH1+2fdFDisYRFNb1gk5yrHaV3lOA6z7iIQpxIzg3p83EQgLy'
                                                'xgBjXOvUaBc6/JYncTg7oPv0DDh1+QmcsCeNayAK2vLSMQ59cm'
                                                'cf/XJunMZVkE4jx+NIM2/esbAmn7yIVBwxcbCWTFiUcE0mjnZO'
                                                'qwc7ImG/cRSJPfMGnxG0Zp794ygTX+H1YByuTuArRNjkqKHKXK'
                                                'G4cFKPYbJwGKmPq3AEW6WUzxbhZrUuMgBd7yzGQC6dVnjNr3Gc'
                                                'NRs7fSgtlb5YcLkzTtwiTeR8doDx2TVR/6auyHvux99BRNPHpK'
                                                '/mdv0C57A/e4Wkg9rxaKbWSgAtvNqaRucypl1YUgjb0QxJ1Law'
                                                'kk6h+TFLjhs18IZNRBHwXOHnSCjg06IfeH+imwwxihPmNEpiz1'
                                                'V+Dp1nwC2TjMT4HTxuMoO17kl2X+mrXMn8Py8wkkZqdBgWNvFt'
                                                'Oam8Uyd1+AAu/7rpz2fFcu2akGBbZrKSaQ9fsDNGF/AOfuKifZ'
                                                'hY/vCFRg41+nCcTwcZACR/Y8QyB5OwM1d2cgdzxVRSCjVgQpcG'
                                                'avM3Sk1xlZ7Ie7pp+RjUVnKbDorFxvwtmlycgNQecJpP6XCQoc'
                                                '2XKY5rYclr07J+q+nRM51e4IgYxcNUndV03i0PtZNOV+lvxvio'
                                                '92TfFhRz1OIAX/8NXCf/hyVvRJOhJ9UoYU+6prsS/ndOSQdORI'
                                                '8ko/BS4MVyoOVzkyxF+BQ3bn0ZTdeTI4dpICJz/Iom0PsuTGVB'
                                                '8FNuYfJ5AfHHw1zcGXMz/E+B+elDsY34bxU37LIZB3VvlpxCo/'
                                                'jntHCWTpC/4K3JCeRyAZOf4K7HGmgEadKZCk1w0KnHu9iEA+fd'
                                                'ZX457F+Esx/tKTIiW+ChzzMIeWPsyR4bF+CuwzV8l7rornUH8F'
                                                'jtmbR8v35skcq79GWP055WwBgUx6w6A+bxjYsamInJqK5IXrBg'
                                                'X2vFRCXpdKZNMHAZr0QQB7HC+jMcfL5DbGb8H4TZFKILcxfgvG'
                                                'tzuQR90O5Imj+Cuww/kCArmM8RswflRzEYEIxgeObCjB2bhERk'
                                                'QHqHt0AIecKCPLiTL56HaAxtwOYPPaCgJxiAxU4ATP0wSSj/GB'
                                                'E+oLaF19gXw+06AJMw2c3VJEIHNuGDTihoFTrpYQyAmMn43xw0'
                                                '+V0exTZdKtJUCBzfEYP75CkjE+8GKv07TI67RU1QYqcGdxFYHY'
                                                'c5ACd0yrIZDHGB+w3pVQZWOJvLYwQMMWBvAKaxmttJbJFoy/5c'
                                                'n46yqI1lWI63uBCjzd+zTN8D4tM84GKnBOaRVJaZWsw/jA6dNr'
                                                'aM/0Gum5J0iB7axn6K+cMzLmGaMClw4+SyAvtAYosDGhgkB2Y3'
                                                'xg74mnaeLE01KD8YGTyqtoc3mVPMb4wOEzamj2jBpZg/GB0+QM'
                                                'gWzD+MBhrmcJJPV9owKHR9fR7Og6qT1pVODkH89Ryo/n5NU67L'
                                                '11gRxeWUUg/2cO0hXmIHYOqyHHsBqxYXxgY+4ZMueekR72uFvb'
                                                'G7l+yFm6NOSs9IjCr6OMXLmwjqoW1onPKaMCP04/R4/Sz0ntsy'
                                                'YF9qk5TyDDIkzqFmHiUNsFApnwp7d6/+nNdWsy6PyaDEldNUGB'
                                                '3Z8/TMOfPyzZdycosPHn/xLID/Mmatq8iRxh+pnCTT/L/LMTNe'
                                                'rsRM7UI3REsV4EY70InsSJhizaaMiSOwcnqe3gJHbYd5RAzvXz'
                                                '0bp+Pry4z3EC+S3WRztjfdj2YTbd/TBb3rrso+GXsf7mnqCc3B'
                                                'MyMMhXXYJ82eHZU9T72VMSkuKrwM6v5ZDTazly976vAj9OstJf'
                                                'SVYZ808/BbbLEwJ5epef9tjlxyF3lELvqGztxHrU6cf2TnkEwv'
                                                'cmqPneBHb45b9k/8t/xf39iQrsyj8TyG30taAvJu8IgexGH7Ah'
                                                'MIv8ArNkQcYkBc7cf5QO7z8qrzj76HRnH7Z3PE49HY/LB6t9NH'
                                                'q1D8cuyaY1S7KlDH2l6AvJP0Eg69GXgL7EPqcI5DD6gONfz6F1'
                                                'r+fIm7hrAEfhvgGyGX3AkflCEfkiC9EXjb60u0ogD9DXib7wfn'
                                                'kEYpnir8AZ3vl02Dtf3t/qr1Fb/blxSgFdnVIg0XUTFdij4AiB'
                                                'vGCZpMBNQVkEch19jegLO3iUXj94VArQV4i+tL7HaXff43IZfQ'
                                                '3oc1iWTSABDT5qaPDhwoITVFyA/RN9wD0cTxGIy7d4nt/6ctcb'
                                                'OfTwjRyxoi8HfTX/tlLlv61yH33AOQVCUiBSjj7grntKIDMf+C'
                                                'lwunMe7XHOkx/Ql4a+rgn59GhCvuSgD9g1FOe50ALJuYFf3/Bn'
                                                '41uFRG/h/jLOoMCJ7xYRyDuHJmnEoUncmXGUHmYclcD+Pgps7H'
                                                'ecAvsdl1fW4Hmu8eGUmGwC2Yc+4OlFJ2ha0QmZa/TVuf//3/uv'
                                                '7XtK1qEPOGpWDi2YlSMjO33VvdOXDV9bye9rq7w13U/Dp/uxa5'
                                                'HQC1i+PXf7KfD0X5Vm/KryC/qy0NfZP4/u98+TP9H3GH3mSfkE'
                                                'MnCbv7ps8+fwlwto9ssF0u+mvzrf9OcV4YW0PLxQ7LwMCpwdif'
                                                'U7skjaVxrUttLA0+cVE8h1q0Ebrdgv5pTQ3Dklkoe+XPR5LM+m'
                                                'McuzZdAVH3W54sM1xSeotviEVKIPuMvpFD1yOiU29AE3vJlDl9'
                                                '7MkST0Aad9YyWQU+jLQV9csdDaYpEk9AGn3cf8vK/S76GfOj/E'
                                                '+zgwjywD8+Ttqf4aPtWf433yaZ1PvixHH3DytALaNq1APkTfYv'
                                                'RlvF1IIG+iD7jlvSJqeq9ITKsMalxl4Jj3i2n5+8UySbD/ioHj'
                                                '55bQ2rklEtw9QM3dA9jzjVLyeqNUYqcEKHBGcBkdCC6TQJOvAq'
                                                '9wPkUrnU/Jmzvwvu3w5enhOQTSiT5g+2Qr9U62itsrfgqMrZZA'
                                                'OtAH3PWb0qPfVD5CXwz6trjkEchR9GWhr9Q3n0Aq0Qdc/88COv'
                                                '/PAilAXyH6GmcXUsPsQjmIvgz0Ob+P88P7RfI1+ragLyWqmEDW'
                                                'oy8BfSnvltCOd0vka/RtQV/ozFKaMrNUCtEHnG0po5OWMhmXGK'
                                                'CeiQHsPracRo0tl3MlAVpXgvPnsxV04NkKeeeBr0Y8wHlnq5WW'
                                                'b7XK5+hLQJ+5TIjKRF7/0U+Bwx4ogZxBXw36GgbhfDUoT5xC/R'
                                                'XY0T+fQEZt91dgx1cKyOmVAnFt8ldgx4hCco7A/ByP+TnewOao'
                                                'IqKoImlCH3DO/GICOYu+Onkyf0swf0vkGvoa0Rcxq5RAnKYGKH'
                                                'DlS2UEEos+YG/PcgJxKw1Q4Mw+FQTSt2egOvYM5Kj2Copur5BI'
                                                'S6ACJ+ZX0sb8SjmCPuD0h0og3v/zU2DP5/MIJAZ9wOGGfAKJRx'
                                                '/w9BkFBLIcfcChcwoJ5E30AUfPL6LF84vEP9aghljM1wXF1LSg'
                                                'WEaqQd3VwPXvlRDI+B4B6t0D5503S2nlm6XyLvoi0dcQUkYgee'
                                                'jLRZ9hXDmBfIC+6Cd9z6HvuQp5G33h6Iu2VdACW4Wkow84vqCS'
                                                '1hVUyuufBiqw29bTBLL8WKACL36viqLfq5IK9FWiLyUgn3YE5M'
                                                'sl9AEnvVpAIBXoq0Rf/NxCvE+FkoE+4NQFRbRjQZFsRB+wQ3Qx'
                                                '2UcXy0r0AXfNK6GH80rkU/TFoS/xrVICOYA+YNvkMmqbXCYOXw'
                                                'YosNmrnCxe5fJf9GWiL8uxgkDS0Jf25PndqaDIOxXSgj7gFYWV'
                                                'BLINfcCO206T07bT8gv6stAXPq+KZs6rkml3AxU4anQ1LRhdLQ'
                                                'vcgxQwH6ppXnu1jLjlr+63/Dnz3UI68m6hPOWN+elt4NIPiqj8'
                                                'gyK5gr4G9HkvLCavhcVSij5g56gSGhBVIhXoq0Rfengp7Q4vlS'
                                                '70AdtNLaPHU8rkLfSFoy90fDlNG18uXegDzu5bQcf6Vsgt9DWh'
                                                'L+JuBc29WyFeLwUq8OKiSlpUVCkX0VePvsfos9t+Wn5DXyf6jO'
                                                '9XEciX6EtEX+iYagJJRx+w0VZNZlu1zHg7SIFD9tcQyDdJQQoc'
                                                '994Zin3vjExYbVDv1QaeuaiYQAbmGtQl18A+80sIZNDTAeryNJ'
                                                '7H26WU/XapTA4N0JDQAHYMLSOH0DLZhb409IV7l9NM73IJLsN6'
                                                'W4bf74Q+J/Q8g55n0HevgkA+Rt8K9EUWV9K84kp5fi3O82sD2b'
                                                'b9NIH4Hg9Un+OB7BFVRSCl6AP2GFtNY8ZWy3X0NaLP7U41ud6p'
                                                'lk3oS0Kf64EaeuFAjVSiDzh03hkCyZQgBc7tX0sg5ntBOO/hvF'
                                                '5YSyAL0ReNvprZpQSyBX3Ari+XEcgd9NnQFzGhnEC+RF8i+rL7'
                                                'VdDJfhWyAn3AMzsqCCQbfcDhJZX0ZkmlvIu+SPQ1ppwmkJXoA3'
                                                'acX0Ug9vcCFdjRs5r6eFbL0JFB6jYyiB+jD6QIfYXo60QfSM+v'
                                                'cJ/5Cr3vn6HB75+RVvS1oC9pQC1tHlArsegDji6qJRB9wai5L+'
                                                'B+sOwsgeRPNypwTf86qutfJ/4bA9SwMYCjJ5bTgonlUos+4Bzn'
                                                'CgI5ib6TT/p+raCwXyvkMfqAp5dW0rTSStmFvjT0VX57mkCOoy'
                                                '8bfV3oe4S+UPQBd6LvPvrmoC8CffV3q+nS3WrpPjtIe8wO4sqD'
                                                'NVR1sEbM6AO2oe8u+oZrkAJHDKyl2QNr5b/oy0Sfd3EteRXXSi'
                                                'f6gLfEnKXNMWfld/R1oW/LgDr6ZkCd5K1G/2rcv7LraGZ2nbzz'
                                                'k1EjfjJyx6xz1DnrnHSzD1Tg6fcraMb9CjGHBCpg/6skS1ml3E'
                                                'RfE/qyd5ymkztOy+/o60Jf04IqurGgSr5AXyL66sZVU+24atmJ'
                                                'vlT0Zd+rpmP3qsWEPiP60jNqaHdGjXyCvjj05UadIZAI9AF7uN'
                                                'TSKJdaaUEfcBf6HqJvzFDcV4caeeZHZ+n1j87K2FeM6vkK7pcD'
                                                '62jGwDrpRB+w44k66nOiTr5CXxL69r15jkD+rDbq42ojGzrOEc'
                                                'iM340KXP/5eQLx+CxQgTN2niaQwOxABa78oIpAStBXir5sr2oC'
                                                'qUcfcHJHNYF8gr449MUdqiGQY+gDTp5/hr6Zf0a2oi8ZfZ3oAx'
                                                'nagfndgf6SWpKSWnkXfZHoc1x+lpyWn5VI9AE7uKDHpU5eXGNU'
                                                'jzW4r6OvGH2CPuDwt87R7LfOybgafD1qjNyCPpBP0ReHvoQN5w'
                                                'kka5hJgcNcLtAMlwtyOdSkDaEmnrnnAoE83RGoPToCOXV8NYEM'
                                                'HBWkLqOCOObXalr+a7UcQx/wzMM19PrhGrmHvg70RS04QyCV6A'
                                                'POHYT1ZVCtvIk+4ITSWtpQWivb0ZeCvnr0gSSjD7gUfZXoewd9'
                                                'EeiLP1lHIL+irwN9zuHnyCn8nMxFH3DKr+cIJAN9wN5fnCevL8'
                                                '7LTfQ1oa8TfffRZ/+ySYG70AcycplJ3Zehd0I9hU+oF042qTnZ'
                                                'xB7Z9QTy/wBT+0nD'
                                            ),
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [10, 340],
                                            'zlib64': (
                                                'eNodmnd4juf7/4MgiIggVhBESgRBhETkFZGdyN577+TZM8/z3K'
                                                'qkMYqmVkMVNVIfYqVWUatqNWZtqkaK2hRV5Xd9f3/kyHFcOV73'
                                                'fd3ndZ3v8/12KPMu4K9PBxDiJKdptT8bm0upC4tny9MRVPu2oe'
                                                'NkGQcUmWx8NAnc7flKpqL/okKu/hjBT1cHEfSNljHvylkdnMzA'
                                                'Lp5EjK1g/8x8GrckY+UaypkmT/450w+7ndZ4HOxO0ukqjn9STL'
                                                'uvM6j/LArTbF++vuaKpdiOtcM6cbeXAm1WGQUfcmgVk8CLC1M4'
                                                'ax7Ftcc9SVS0Ym8vFTljK/F4UMC1ealcnxfOsXXjuTNqIETY0O'
                                                'uSGqstMpbkldAjOItDO2LoFQ4xjm4UPe1KbbqWqhsKhqnLiTLn'
                                                'MS0qiQPZwaz4Ygw12X15sVGH3xEVlk+qmDKpiNVp6bTciWRpjQ'
                                                '8Z011YfFaPrELDk5ly4ueXcuKPbDIuxqF/PZknk0eQF1XJjJml'
                                                'HLIrJOFCNrMPpjLnSBymM+G8+n0y2W/G4/TJCFbPHYg8rgdxjT'
                                                'a8HGiN8w17vpvSD+WiKp7ZlHPhYhGhD3L56JbBudpE9r6cyuj8'
                                                'IL65NJHVGaPZ0NGVvPZ9qJvRGd0+K0o72xK4rBeNv8jQqSu4+F'
                                                'kJDrfy+UyVxZthKSx+EUP3Q6EMWwT/KsfxrsqNM9/3536MA8nL'
                                                '23LJsz2xf3RjfIuc4OOVvHcs48buQhyX57CrIY38X+MpfR+B+6'
                                                'gpLCnxJmDnSE55DeZZl57Y6TryLKk1//bqwqAXCmbZyNB/Wc5F'
                                                'XTHNa/N41jaThhlJ+DhEY782GHffSXz6xxjW/u8Tyvf3pUt4F7'
                                                'bVtqY+qCOWR0oKxsqRXlUwsm8px78soC4gG9veqRxpE8e012Hs'
                                                'euTPP395cbetO2Gpztzp2h1NUXvWlbUl6qKKF+EKfCdX4bS2jD'
                                                'G5RWzIziVtTjplZxPIGzwViyEQj998SJriwdYbLow+2Yv9frZs'
                                                'nWPF3C1qQmOUXNXLOOdYgU+PEiKz8tFez+RIdTKnh8fw4Y8Q3i'
                                                '/zo1ueJ23ChnFK2Q8bu65sjLdmqElDVaAK43I50xWVdN9cykoK'
                                                'yeicw2dt0uhuH8+7IRHICECTM4GWpSPY2XYQd8/04LdRHbD20V'
                                                'I0VM24TQpSdlTxalQ556yLcRqQh6k0gw1nEikLiuLe/iCcJ/ky'
                                                '+eRoRn7qSqcZffBqa8e7h1pc/1PTa6MSfYsMu68rcNhcwgWHAl'
                                                'Y0ZjGqKgWTfyxve4dh9QYCbo5jw203sl0GEHvGgavzdSw9rGFk'
                                                'vYo2bRTM/6uSJZTR8KCQx+dzKHyQxryeCUgxkQR+MYXsM97c/2'
                                                'QUbxoH8/MXPSkbpme5SYtMr6ZzZyVrhspYuLOcU2uKMV/Nw21K'
                                                'JkNPJjEjL5rsj8GkrJzEw6ixDBk4lBQfJ7J36ekwTMeZEA3ftV'
                                                'VRFStnQq9KqoJKkR8uQGfK5mV6KmExcXyIDGf01Mm8SxyPSe/O'
                                                'x3POVEwy0HBch1cHLcceq+hQrOCHvCp+OVXGs/lFJCzO5cCJdC'
                                                'b2SaSrdipjLgbSMHEi57Z58GfCECbsNuCTrWfaPi3qo2pqy5R4'
                                                'LJXRwVfcx8klXP0yH1WvLJ4cSKbJGEOLTyinPvjh3+yJcd8wnn'
                                                'WPxklbwai++RzpqWGczJc94+Kp0OdTEFdF6v/C8a7JxK+ijIXH'
                                                'lMxbkMTYQ4X4/CqjwE2LjXcOTUJD199XEfmDntXzRxPaA4rehu'
                                                'MxPJHv9mYSeLqA4AVl6D1krNnpTXx4MFkjY7EuSaPFIY8TKSXs'
                                                'L60kRKvAPX4ykQ8ieXg6CY8+2bg+Fj2tK+fufRnzFSrs0kP58D'
                                                'yONrfTeRqWj1tiKWM8q5jQU8mvbhrKTkUJbU7hZHkOdVbFtEqs'
                                                '4OHXchr+U3Hqey39GhKoic7kD00BrUPFPXldxcKtSqrXaFj9t4'
                                                '5+L1Jp2prLmQ4lRLlV0jxFwZPPRI9107Glu4EjW7JovaSQxaPL'
                                                'KdgqExqgolNvLU9i9Gy0NdLq6HAy2o7Fu8WbDbn+DFMFo+4VSU'
                                                'xsLIEDkoiemcaCz7OwGZ5HrKaQgooSCh3LMRoq+d8yGe97eNDQ'
                                                'fzyqI5O41SqQw+fDcPSOpnFMAu8OpbDhcQZVO3NYP6SAUX7F9G'
                                                'lfxuKaCnYfqqLqoJysYE8W95hIiWUyu0whrLObSqRnHManSRRN'
                                                'SSfLM5uaX/KobV1E2V8lJNWWU/p7paiDjCvnFZQmTuDpfT8yug'
                                                'Xx8Gw4jwbHUNtR9HBtKtpVmUxLy8V6dwF/Hiom3FJG3PMKegyS'
                                                'oXRU0PaCkm2TfTmyJICL00MpahVFS7d4ivcmk/IhnTE3s/HKzW'
                                                'fwF0XYlJRi+7acmClVnIqW4zNYSdvDKhw6+lM6MJhvTkWQ1imW'
                                                'Lr8nsj48Da+4LJre5bIlrJBBQSUMf1nGq9RKaiQZu0sV9Oqv4t'
                                                'YqNTs2TcEyM4ydz6JwfhaPfU0K7jsy6DYzh0FvxX3pUUzg7VIW'
                                                'FwqtXFvFoPVy3sqV9LFRs0ChoZ17CDcOR5L9NJbgLUk0dUjn2c'
                                                'csbJfm0XSpkD+PlHCisJzPDgitvCYj9EcFK4tU4plq2o3V0jQj'
                                                'nPrr0VgdT8ASmcr9skxshuUSN7eAhvpiXNPKeHtG9GZbGY3v5f'
                                                'Tap2RvhJp5mzS4/qllw86ptDoTxxhLMst/TmfjhmyWjsmnT36R'
                                                'EL9SRl4sx2ZIFcc85bh3Enq1SfTdQA13yrQcn6fD4ecYZq9IpK'
                                                'lLGoucsuh7QtzlfoWs7VfC3tNl+PpU4pkvwxSjYF17Fc+/UBN4'
                                                'X7y7r459bnoCNsVzNTSFPEsGyrgc0k7k8/CvIjz3lZITWMEXn1'
                                                'exYa6cklQl7i9VfMjSMG+VFrefdBz5Uc8ReRJ7Tqcx8WQWcSV5'
                                                'rN5YyLAVJXScUk7FqkoSfpKxYLWCtTEqDjWr2TNQS7soHXuT9S'
                                                'wJMBBgn4q9Zybhb3LIyS5gpFLM5HFllG2poNf9Kg7eltNunRK/'
                                                '8WrylmnofEmL9ERH/T09d38ysHhuOre02exok8+FCUVMcirFqa'
                                                'mcBe2q+Lq3nNdi3jsvVTHWTsPrVC3Nko4Dn+kxlhqY6mlkY4vQ'
                                                'uq9zaXxewOO/i+mxsYz4PpU0BctwHK/A7rmS2wY11Rc1FHTS8a'
                                                'uTnho7A23vGqhYaWR1qFL0tx7nA2M5+iwG70Fa+uzVM2mDkaE9'
                                                'TcLrFeFgW4njfQVuX2lQ3gzE3yYOhxMZeKcWss9si2dPZ/Z4ji'
                                                'FtQgCL1+q4/JueO9cMhO8z8sJSjccQE8e2m/gwwswHnRKrDDVu'
                                                'H8WzSnW0Xq0neLOBJ4uNWGVVY8gowymykmPXZfyvl5KkDmqhwR'
                                                'oah+s4WKEnq1sW/+jy+Dm9iMDjpfidrCCtQsatHQosTcI7p0XS'
                                                'vS6OnaNSeD8hE+sduRRsKaR2TCmjQypwnuFJY4Mvl70COTE0Al'
                                                'dzLHleyXxIzeDQU6FTMT1p9dKZnmPd+PBuDM3eEzl1J4DgF2E4'
                                                '5cdwaZUNN6tbcWVhJ1xm9WCZzwAC5w0lrXI0Nke96RWrZ/FtPf'
                                                'XifK62GLBON7LjqJHZbtV0+ryaA9ercXE30aAxsXe3iUNvTfQe'
                                                'a6ZXqRm7ZWYWiZ7c8JeW7Yt1zB2nx/q4nluxBq6cMbA82MjJ7U'
                                                'Za967mlLaa7b9WM7a/iQMlor6bTGx/bEJXqyJBrsYxVMNBey2R'
                                                'J7Ws0utY2VtPn816LngbmL3HwKPRRr4R5+/TvpqEomrk+6uxtz'
                                                'cRPErMnlgFBTlK1maoqA1RM2iQBu1jDZYNWoam6Uj4T8dvC/Uo'
                                                'B4rnfGfghZMR9y+MeP5t5KfWwhuXVxKzrwr1Wxkz+yvY46HE30'
                                                'OFZ381jf+qaTquIWCWlqCJOtbc0uFp1LPXxoDVFwYOXCmieXEJ'
                                                'VhPKaNxfjod7JZc+reL5ARl7H8pRWykJbqMi6ZWKnRfUaNdqqC'
                                                'vW8k7owVeHdYyZlcOp9XlErhPzsqaIgNASip6UcsxQzsG/Klgd'
                                                'VMXuWhnhO+TMO61g129KHh5X4btJzU/TNEz7M4VwtdD+G5kkDs'
                                                '7hdkgefSILiBhdxLG3xTz8rhQ7z3JqN1TQaFPF6WiZ8IdyOqxQ'
                                                'ULdeyYqz0cwQeeVyUyIhlSk0WqXTWp5Jxt5s7rfkcvdZPvXXCs'
                                                'lpKGZuSinGx2V8WlLBmxOVuDvKODg6iP3bQwmyE5o+OZr4qXFC'
                                                'TxO59DyZ23PS+F+rTIITsrk+I5d1C/N5OqMQm2Qx22zFvtaV8X'
                                                'c7b1a2mUTNVX+aZwfyqFso9ZoILm2O4vqhWHptTsBdk8zdHmm4'
                                                '14k581cWewfmkj0+n6ARhfSrGM6mS6MIHOSJTfgEqqb64jvcn7'
                                                'rbUzitDGHttXDa9ROzanwsKW4JhL0RM2dFKjMGZnBgWhZTljux'
                                                'LGcgNrauNC5z4z/bUWzPHMufc8bzftFEGo3Q0XsKRWeCKfAP50'
                                                'HtVPpsiCFhTTzD9El0eWHLlyJH/nfTkca9fRk305mscUPwPzGM'
                                                'iZNH0mrpGBaf8iLuhg+/H/PDd14A292DufhdGJ+9jKTdgDYkPG'
                                                '/FAKv2xHew5Z1NV7z/7UHh9T58bBjAr5kuXHgzlBL1CEacHU1P'
                                                'Wy8qBvvQ3MMP19uTse/ZDYuTHUcdO3C6gzUF7VpxuEc7Qpw78W'
                                                'KgPfcde1D1b2/anOpP2szBqFyGMnedO34dRlMVPo6JB+NJWiTm'
                                                'tpWWq48NOM7yZennoZxoiaNqczpz+2SyZlgBG56WMqNe3J8kGR'
                                                'WPxVw8qmFEOzGPhDdeaG9kQn41unIT5rSRrEsaR83fE0kcHYCx'
                                                'nfApqqkk6ePo1TOZoqAp3GgfRvu8KCoS4unzh/AunTIouJTNiP'
                                                'B8JoxI4Mn6FO4fyGCGKYdzv+cT/qyIhu2l7B5VwYn+uRxXFvCP'
                                                'uphew8X9qa8g8kQVW/fL0ZqVBKwso491JQqRM5odFJiFp9ZHqS'
                                                'n5XsO6G1p8ChW87Sq8zHzhCe5oCOyq45mYMW4dDXheN2B00nJi'
                                                'io49YXoujjYQ89GAww9G9iRWs+6O+EkReWKgEdl5Iwvk1aT8W8'
                                                '11oY1v/zARO9lM9+2ufFnshp3NSKK+Go2vzTj+KJyA/H8TmXzZ'
                                                'j5N/TWbDvUDqfw5BOyccr/FTeXYsmlP+YkZ9myBm8xhurhkHHy'
                                                'ZQjC87iuF7bQC3y4NwCQklyiaC7M1TqfSNYePmOLI7JNISlsxG'
                                                'RSoen6az4JwvKa9hq9UUAp8HcfGXUKJnRLDZNQrHzTHs6BfPda'
                                                'VgGpMJPpeK6UY63c+KPL5J9Kgql1OaYDaHhKH9GMHQb6K47xzL'
                                                'k9nx1F5LxKlbCn3GpNHWW3jkoVk8s8rB43Aul2T5LGlfiOucIk'
                                                '6ITN/yZSwJb+KFH05ipjaFIV+mceqrDGylLKbH53C2ax7f78ln'
                                                'T1Qhe5uLKJpYgnJRKUtvlKHdkITLlRTKnqZxVnjxRxeySFubw6'
                                                'HsPM5YF+C9qJDtXYv52yj81tlSDghvvze8goKqSuTTq3AMycTe'
                                                'LRvT6xysNgldjSrAQehXh4RiHu0uQdW5jNToclKnVeAq/JJ+Wx'
                                                'X/NYms9b2cpAUKFvyTR/8lIk85F3FokdCptyV8LXJMyeflyH+o'
                                                'oOR8pchGVbz4Q8bt83IMuxRkLVAyXMweyUnNzYvFOHgJH/lZGY'
                                                '/EPGj9sAK3tmJfXWWo7OQ8/0+O5y0Ft35QslBS0TJJTdlzMWfq'
                                                'NfiIrH/373K69hf51quKRmRM95FT94mCmvbCh1xTsny1ipdZat'
                                                'FzGux3afgrUUuDyPhHtDq2v9NxrlLGgCI5x4T/VI5Usv+jktCf'
                                                'VTRIaipHaRj8m4ZYhRadtfCyItsv6apHO1dPXysDjysMvEkQXt'
                                                '1FxdI/VVivVPN8qoaW5xrK52qx6q+j/3odPkP1/LxKzzCR105J'
                                                'BmzvCe8UaOTyN0Y6HRDzt0JDhp0W3/VaAsbruL9fx7d+eq7u0P'
                                                'PxEwOaOgPLXgs2wUjWJiPrrKp5HFXNtKXVHD6oxT9Gh+yijqMJ'
                                                'eraf1LNiooGHawy8bW/EudDI7r1GunepxpJRjf/aalY8qObmMB'
                                                'MTikxsKdMTd0tPu2gD7jsNuPQ2Eqwxcv+kkbR+1dwvFf26tRr9'
                                                '39V89DTRKDexv8FE4Q0Tl+3MTDxuYICbEb+ZRh5dNaIbXo27rp'
                                                'o44TdGtTLRyt+EdbUJ1TYTM1pMlPU0kxBkRiE383CpmXkKPdMv'
                                                'SLz/2UDqdYnGlApMd8XeZklUPJFY0a2KNyPNKBdKOD6TCNggY4'
                                                '7SzPIVEkufSzgNUHB5s5lz6yVevZBwt8vlwxAV/7QYcbK2cGGi'
                                                'hLpJ4tltifOvJD7k5vFin4ruQ6qpc7IQFySesUsi7a5E1t8SDt'
                                                '/nMz5a5Osc8d1jLaRGiff8KLHlnsRt8fc9d0R/XFWzS9S+XbiF'
                                                'F0kSD/ZLvGmRyHktEeFQxBGRQaY1Cx+YbcE5W6LrIbGv+xKXxN'
                                                '97jStm9g0NR0RtJqksPCwS7/xZIumBJGathCGyhLVJWlaMMZFd'
                                                'I/ZXJfHoF4mqhxI/iL8fEp4i6oTIN7kmliy1kKORWHNCwviX+I'
                                                '63EjtmpbLqhyIifpDRZZ2GvGoDKX9Vk6k2oyqz8LqNxGf5EndE'
                                                '/Q4dllh9VeKC2NtqUcue4vlPktMICCompYecBXc1pAj/V29tYt'
                                                'wMM2lqC4M6SiQXSwStkvAV+553TaJI7P294KMEv2BQOu3PFROn'
                                                'krNOZKmsEwb8+5nY+JWZ1hYLrvbiHMtFTb+T2HRUQiHO/JXgE8'
                                                'TZfS74NQ/S+ZBeQuBpOWmZWozPDMSOM3FktZmqzy1YO0r8KGpy'
                                                'ba1Ej2OS8AkSZlGbDYLfK/jRmzLwuFOC9zAF25Zo2dbDSHOkCc'
                                                'tWMysXWNjXVyJcKWreIO7FcXEuNyXaidp9EPxTwQdUCm0vLsXn'
                                                'UwU7zmjp6GNkX76JGz+Z+fprCwkDJY6ImvtskDgp6m71u8RswU'
                                                'e/lESel0RfZSF/IM7okoIqGx2zMo0MMJq496uo32oLv7hKuBrE'
                                                '926U6H1K4rTgOz6S+FbwCYK3v5PFopIypg1XctNP+N1pRv6Zb2'
                                                'LhNTP3Nljo5S6hMYm71iiR/atE/S1RN8E/FvwMwd9amo3zvTKu'
                                                'i9neVa2j02ojU9eaeHbfjM92C6GjRY9JEnlbRG81i2f8Ib7hsY'
                                                'S36I1tgk+JyKFvdjlFzUo6NehwPGyk/x4xr/82k/GjhSQvcV+m'
                                                'i/u6TfTTaQln0VcGwU8X/B+Cn/Yuh7qL5bj1U3Huqo6MO0aqmk'
                                                '1sa20h4rB4v+jHuzMl/ERPtj4rcV3wjwV/UvCd/5EoWJfLvIgK'
                                                'PMpUyG313GxdTeBtE327WLA5Kb7fX/RhreijHRJu5yTq7ohvED'
                                                'rhIHrTW/B20Xk47a/g0x9U3PfVs3pgNd/+bcKrr4XF5yycChQ1'
                                                'nyNxf6dE6HmJQNH3zYJPEnyB4Gte5RE4spIhH1VMFnl0I9XMsD'
                                                'Fj84mFV1dEv4VJNMwTNdwtkSt066XgfZ5KfC34LwR/eXE+Dssr'
                                                'cRW5aka9ng9Cc1/1MbNgjAWXPyxcnSoh/1KiWGiK6jeJ5UJXvh'
                                                'P8dcE3/d/+vQpY0L6K2V+o2faLnpWGal64mzk7ycLQ+xbGx0m4'
                                                '/J8W7pOwXBT7F7rTUehiX6Et1wUffK8v4X/mkH4sn4hU4U9jHU'
                                                'kY34ead1NwH5LDugX9iNMFU742l2l/yxnyRygtPfK5tlOBy3AD'
                                                'NroCXsqVTKkxMOqViaYbI2g9vCf7XFohy+7NrsujyE2ZQkZUPF'
                                                '8fyma1tgeNLa1ZbteXe6mjkZYFYohOYG9NDtLmUlIarIn40Yk2'
                                                'l8bQciWItxGJXA3OxXlSGeWHhYco7M+HaE/mdwvhrvA/Vm3yWP'
                                                'BLGf+6Kni3VejkkXEYQ0MJHJ+Mz495XI8qZ2itgnWvNSzvZmC1'
                                                'PoyBrinUC0/V/Xw571sULPcWuhNgYG6WODv7VNychT9JqKBmip'
                                                'K7Ri2PZQb6WqqpfSDu9akCpp2r4MI3Smr3aIlebuDi8mqaPpj4'
                                                '97mZObMn0MBIukwYzNElPVk5txM+TVYcfW5L4MteeM1w4VfdKC'
                                                'b/z5vr7YRXN0eQ3jmehXtSebIgm5nVI9hUPIj++x0J39QRv0Wt'
                                                'SDvTGc3J3syMG0KBtweds3yo/34K7ewjhceIp9o9DetX2Uz/u4'
                                                'CZiwfy+E4Ptv/agfHa1ixptMP2uz60cnbl3T8e/NhlIsMiArFa'
                                                'FcmqLgmEfpNGj6QcHMMLWTKnFO8OPbj12Abn5DaY5nahrbEv6x'
                                                '65Mn/3aMIPTBTaFcgJj6nYzE/gg206LjtzOPBNIV7nS2kqq+S3'
                                                'zja8Hm9NqzJ7MmOcUO76hP76MeyS+bJ/dhD5P00ltHMiK6pFpu'
                                                '4p7tSzQtr1KePFuko+2SBnW6+2DA7pip1rP7I+G8q6cWOx7j+J'
                                                'Q0ODhS5HkTc7kWFC8wO0uaQFFGGTU4bpdiWzr8uxzlFxfbAD3f'
                                                '/tR+nUYbi8GIv7yUks/jmYjleiSG6XxKrEDJIu5BL3VZHwNmXE'
                                                '96sizk7Bq1kqPl8pfNjp/ixwdGPRJk9G6USezA2hZ0E0bp8m0f'
                                                '5gBhWj8yi4U8SmB2Xok6sQE4m87SqiDmtoNus4fdONt8Xj6DwI'
                                                'tvwbwuvH0Vx5l0TG8ExmfZHH4jHFxI8o59mCKrrLFURcV7Hmno'
                                                'bZ9Tp2tzcQMdCLgyfh9cJQGrQxlKiSWbMwk83383gwrRhPeTm7'
                                                'j1fRbaWCJms1c9trebBDR9kAA3aRRqLU/nQcEcbIDzH89jCZm+'
                                                '8zuRmQT1ZzMVHbyrnYSobltLjz7mraD9Py6JyOE14GarONfJ4k'
                                                'fOLhMGbOiGVIbgq2BVn0Xp7P2z4lNLwsJ8JbRuIHBe0S1OK+aK'
                                                'l7ouP3qQYOKo3MKKqm72gT2zvEkXQxhVEnssh5lc/uohIme1bg'
                                                'KJdR767khVHNnjItNzvoWVdgYLfwgRHaarz8TLgKH1cWnkr4qG'
                                                'x2RhSwcmsJfdQV+KyTEZOmJH+VmleztZx10TNI+BTVUiPXZwrf'
                                                'FGHi1hMTa7Rmcr7Kpv/qArb9W8L47RXsviZD+7mStF/UHN2gpQ'
                                                'LhX7808GSDEd9F1bQkm9D+I94rmcl/YmbpmwKeBYqs9LyCxfZy'
                                                '7JqUXH6kZozwUFtS9fisNzB8n5GCNdWYCky0tDHTtVb4UjHvPG'
                                                'MsuLUEs7smnZvb4F1uEhMUAeyMjuJEegp7G3Po9aMfb/tHENco'
                                                'MqIxi863fOh6LISx1+PYOySDzn94YSv688TdaNwbUyly9idQHc'
                                                'z5uEiu/RjLmK1JZHqlMyomG6W1yO5+fozYHojv9nCWTYphYXwi'
                                                'Pm9T6eiRRZ1VHv/E+bL8fgB9Oobx6IcofnkUT/DWFD5vn8nFVz'
                                                'l8KbRkZffJVCWHcMBlKk2mOHJzk5lxMZ3ef2TjmTuBsPHgsywI'
                                                'h9kRhLWJRWmTxM3laXj+nIVNuhdbYiZR89sULt8LQzMtmtx1CW'
                                                'jTUmm9KhPfSE+e5Uwkp10A3SeE4v5qKt0nxuPeJYV8dQYfRB4f'
                                                'VuLNbyP8OTItmNysSAxHY2nZk4QH6ehf+BE+LoB+6UGsyQ9lb1'
                                                'QEOU5RNJ2K4W12PDUXE/nNI4UqWRrP60QGXp6FaU4ObfPyMAod'
                                                '//7jJBrCJ/OzPpDvZ4UwyByOa/xUTnWKYer6ON4PSSSkNpmm5l'
                                                'QxD9Mp6ZiFQ6sclt3M5bfv8pnfcRJbMv2xXzhF6HEwrt+HMfvz'
                                                'SGpDo5n6KBYXVQK6G0kcGZ7Kiqx0NmgzWaHOxic1l9Ah+bh38y'
                                                'W/Cj5sDGBucxBHz4Uyb3cELdOjeDUylg/74snzSKK+JgXPg2mc'
                                                'vpWBU0sWV8+KHL8mj896T+SIyY8vD0+m4nEg1u9CGPkgnLt7ph'
                                                'Igi6GmXTztPk1k3O/J1A9M41xoBrMTs8QMzKFkUB71/Xy4XDuJ'
                                                'U5f8edIhkFv9Q6hwDGf6s0jGbYpmd2QcWacTmD8umWYpla2N6d'
                                                'w6lInDvmw6rshFM8CbVXW+bHsA3VymkDY5mODJYewfHMmJ+1Es'
                                                'XhBLSd8E1s1K4sS1FIoc0nEamclK0cPbhL73HjCBwfUTCfzHj+'
                                                'sTA1DmBrG/NJSlCRG0GhRF73MxDCyMR3stkTnjUnisSmPCkgz6'
                                                'r8wibX4Oc5zGk/etj/hOPyJiJ+MuBXJ+fgj2n4VzOUXUoGsMtR'
                                                'vFObol0vmLZALOpzLmQzq1dln8Z53D6Z5e5K/2xtR1EgUF/tTW'
                                                'T6FmuzjHzWFEfRGJXVS06PVYxukTiLqdRK5HKm3y07lizGSpPp'
                                                'vH9uMY9t0EhvXx5RMNgg3g8oUgLl8OJe+nCCy1UcR6xjLgcLy4'
                                                'p0mEzklhw89p2N3LwPVhFg/be/LjqvF86TyRrz/zY/iJyZx4GY'
                                                'irVSjdn4Yz98BUtqhj2NUpnv9mJvLyTjLOrmkMnJrBd6lZnHo/'
                                                'hn4rvNjt4sP++ZPQ3/Qn0z4QhyEh5DmF4/s6kqZt0byPjaPoQg'
                                                'JZPsnkz0ily/Z0bH7JpP7paJKWjmONqzeLv/al8hnUuU1hRUgw'
                                                'USHiLg+LJO1JFA8XxTLNOYEZ85KQ3UrhnGM6CWMyybjlQdkCT1'
                                                'JcRQ1WTcTeCioCAjhWLGogC0WTFsHqT6KwXIohvCyegluJ+Puk'
                                                'UKZPY9qyDHqeHkVOzVguDx6Pfp0P+Z39eJw8mbqZgfy3KIQPte'
                                                'HMyprKLscYDm2N492oRM59Ke7ipVRqW2cQ/XkGY1ebcPOO5p+0'
                                                'SYzrncfl5VqOeJt5L7JJ59JetOqXxAIxH848rMaqXQjeKVZMnx'
                                                'NM/9gKkSGV+JmT+NOhN4unjGX24gLm1chRanRMS6kmqdrM/DXC'
                                                'a4tcv2KZyCiiJv5l2ThVVNK4WUNNhpGvn5u4n2rhlpPIAXn9Kc'
                                                'v3x6sqle4HSzEuULHT3cD1cBNpH8288exBQZwDrRd60bZO9Jai'
                                                'kAQbBQ4/6URGrKZ1XQB/Dh1I7sO2fPVwOPePh7OhJIfXbat4K2'
                                                'Z5U5PwU5MieD3TnZL6dnSd4cwT+wBqv0pjrrYMq1EiZ24rZMvb'
                                                'OGxdxuP40QGvWd0ZUD6BR7fieeZn4NkWFRfeluJxLJVrLf6M7T'
                                                'CA0+42JP45gozDhayZVMHlCQpqW2mYJzJX3r8GKhaIzL/XxMWv'
                                                'zViL7ND1BwvBnUVWiBKZ3SxRslLkbpFhIhVpPM7MpyS3DPkpGY'
                                                'ceq1jeSYddroFb/apZW2wiMd3MBzsLPl9ZqHhl4aTIcO5ykeEX'
                                                'i8yfHsGYMckii+dwIaOE6VRRdUDJs6Fadu/U026lEV1HE7Mdzc'
                                                'ReN/OiUuQ4kaM2fyJRnivx59gJ+Iu737w7nrKPmaSOEF5svpjq'
                                                'OgUXPTRYf9BxSmT41mI+Xz5hYsYaM7cDLVzZY6Gmq/ie+IHYdR'
                                                'zLq20B7GgXw/ethOafyce4p4xtYt6bBqrRjtFhazFQMraaOp0J'
                                                'x2IzDT0tLBV5/pv77Xlf2pdV/xvBwLZ+3OwZyd5TyXw6JZe8fS'
                                                'VcmFXFo7dKjPFaai6JPLjTyLu+JoyDzVy9Z6a/qwOW9a24nyPu'
                                                'idNQhvfzZv+jEM5nJeA5I4sri4qw3KrA+QfhK7M13HHW8/MoI8'
                                                'E7q+l+xcSug25YJvWiYZk1OX5duJozCLf/xhI1cwryHTF4LUgn'
                                                'bnIBvg7l3I+Xkyu82sYUHfPqDdSFVrN6sj8KHw/8J/Rn+JWOKD'
                                                't0YLLFiY/rR+J+2w/tuUhKC1MI3pcrtLcU6/NV1HqoSDBraX6t'
                                                '5+9TsXx6PhDTtHE42LsgdejKp8dbYzY6EjxwGEusfQjYFsqTVo'
                                                'kEtc/m9vsienlUcua5Ao8vNLgmCU9RkEjFsDAUC3zw+WkYCUN6'
                                                'csXQBrsMe/IqBzP7iSc/JAYyviCWf5wyaNxSQFNJOY6L5RxMLG'
                                                'V2uzwyD6bwX8BUGouhyX8UB4f04/muDjwY1pGqL/uxb/UoIvdB'
                                                '9zlT8Xkv9GhcHknGUkJMct4Fl+M7s4COLelkOccSaR3IKKFxCd'
                                                'aD8XzUhZK/2uD5VU8aXdxwe+jDKU0Yg75L5NvabKEXGoZcVWDX'
                                                'tZItF4pwvpBFwKEEnApDcdngjd+PQwmxdeRecmscDV35Wu+C68'
                                                '1xDHEJwtY+jm8e6rGu1HK1jwrjjioq3pRw5Itczg5LYemsSH6d'
                                                '40ds0EiGODiRsNSGd2GdMH3Xn4vLPJi22J84/2qU8w38Eqaj1l'
                                                '9Nk7ec1s/KCO5RwMKUdM4rYhg0dgqDVoxl6puBhDXb4dq5LR83'
                                                '9OIr1+Hs+Z9E42s5A78Ufr17R3KeSqjnSjy/Y6L0TDlzD0nMTb'
                                                'Iw67qa85UBTOgn4fa3gUcD0oly6kjzARNJ78uQjvZjdqsYVryV'
                                                'qH0k0XBKokkv4bPcQl2miTsrtXhsLKH5ucT8qxIx30gEvrfQP9'
                                                'lMXW8DSpF/or9JoumuhM0usYcACX+hGRefiH6xVdGszMF+n5j7'
                                                'JyV2aYT+LLQQGWbigqhdhaeo2WdhpJb2YLDQ7LmvhdaECu/+Us'
                                                '8KzypWXklkRtlQTN27E+onkdDGwt5zRrS/KGm4mM3Jld44W1uR'
                                                's96b0rkWlniKvvLWMn1PMS51ochsu/NCMRS3m4m0FnPm3Ck9HS'
                                                '9VYtctkfOun5Cp6EHkrDA8x5fQ41+JNqIG7V9KPHkoceyGhPyY'
                                                'xKm1EvWVojYOEi6SmC9fmpnoYOLUJgOFdzRUvJXh9ryIkf9IPH'
                                                'ol0fqJYG8LrT0j8XabxOvpEkPGSTjvtKD9y0zydyayFhmxitXx'
                                                'fpQSc0QZvYRezHktseaZmIN/SpguSUzYL34WiZrFSVj9aeGXCR'
                                                'YODjBTllHN7Ll6njmpCfetpP5qHpZT8dx+Ic5N7LtY7Pv8LxLN'
                                                'ayQ+lotv6SLx1GghfbaZ1PYmlMsNOBzTMLRZxjdbi6gPTaVmVj'
                                                'BnHktoxb5txL4HiH0v/VTi4RiJz7dbaBB617pesDVGirzF/Ouk'
                                                'ZGW3MsLHZNE6eCrfbPTi2P/9e5TYt7RP/F4ocS5G4vEdC81jLR'
                                                'zqKc40uhoPvR7b/1Tkd6rEqi6PQ8IvGhv8WKxyob3Yt9f/7fs7'
                                                'iewyialiZjVqLUz7zEzK+2oezTag26gRPSRDUhQR+Eh4VNtgut'
                                                'SMoLvOnumnxb3cKuE5TWKjh8TdzRaO/y7u6AITOToj8wbqhB9X'
                                                'EHCtlAu/Z3L0fCTNvb34GN6HuGxrjuyVuF4n0SLm5JlbFo6NtL'
                                                'Chi5m1AdV8m6/nssj37a5WcCw0j/APcdwZ4CeyxmAu7m3L+3e9'
                                                'OLFaYm+JOMOOEnlKC1+ZzHg/r8bTYGDJAg2lehlnPYrwnp9Czb'
                                                'wgAh64c/hoF1L7d8E53p1mi0ThSInxGy1cvWLG5nMTg0qM6Dvo'
                                                'WHVAQc7qUprmZnIsJ5KYmnF4fdtbzIa2vPAYzP3lk/g7Unz3DQ'
                                                'tNwyyY24scO07MzSg9sXtVeDZU8OJDLt0Xx1G3eBJxwwZj1bEt'
                                                'g9b3xvmLcfQqiuRVe3FuVaLnNcKP3asmpthApVrDo0gZdU8L0Y'
                                                '9O4c8uQVyPcMfbsQu3m7tgeeZO0sIgZn6VwrQGse/zZgrMJvan'
                                                'GBn0XMuxegX/lpRyf0ImHR9H4PrCE9uXvej2c1uiLw7Gw9UPhz'
                                                'bxDI/Mo85F9P5HE/FDq7nqrUe/TMVDTQXmplxcRsRRazsJ+WeD'
                                                '6JlsTUtcHxY6e2F3JZI1dzK5caMUS5WZlVer8Uk08C5Z+IeBMg'
                                                '5sKqTN+WRqFgTy+6bhyAvt8JDsmfLFCEY6BFP7LAWtugjNWhlL'
                                                'xivoeVvF7720rHygo+l6FRN0chY/VrAwQNTwqQLbQBXDjWo6L9'
                                                'QwrVpN4CINgxZpiTLr8F0sflt07IrRk9PFwMV5lSyVVVESLOO7'
                                                'bnIaLsk5W6dAF6Tk6VMlaSEy5naXU3NZzpqvFEwOVrLlmZLQr1'
                                                'Skj1STf0WObqGCjiFKlM+VWC1U0XGUmpuH1ByP1gh/q2T0CyW7'
                                                'xPousT7jsJqSGA2NFzWsTNIiX6SixEON+xE17WI1BF/S4JGsxf'
                                                'asln5BOs6J9XVi/aZYPyDWF4n1lWJ9YJOO1H56NJc1xKZocT6n'
                                                'ZXCwjhVi/bRYf/GpnoA/9DSI9XVive8POoL768mfrqdRrGf4Gl'
                                                'hXZ8Dv9wqOJ1Sy+5dK+kyo4s3aKnK6ysg0ynh3S0ZgkJyQ9XJs'
                                                '2ytYXaDAXtzxvN5K1siV3PpZie8xoW2C066r4qrgfhOc7A8ZOw'
                                                'S3S3BGwdkWKtAL7pjgbBRKAo4qmdZbxekyFa8El+ggY2q1jHuC'
                                                'cwuWM7RBzn3B6QV3S3Aj+yipUPzf/9EQ7xSce7mKL/eo/v//wf'
                                                'xZcGm3ZSwX3BLBJQrvf1twoQfFfRDcGcFZiznkL+Z/reAeCU4l'
                                                'uE+ET3NYoaZLiByb7+UcFlxUkYJtgnsnuDFKJUWCWyG4FsGF/q'
                                                'jilODMgisRXF2LGpsR4qw6KNgluD6HRO/3VbJYcMcE95/gplSo'
                                                'WCU4lw5qLgjuR8HdE1yK4JyVGgJ/0DBdcL8I7p3ghh5TktlXxb'
                                                'eCey84k+DcEtXYf6vG6081DYKrElyd4Pq80dB9vJZMlZLZgtsr'
                                                'uPeCSxI6ckFwnwquQHCzBfeP4PYK7qbgigSXIbiTGi17tmnp46'
                                                'QitVLFVsGN7KjmkuB2Ce664MJGauik0jB8h4YtglsjOAetFqvt'
                                                'WnRPtEwbKnz7PhU1ghuZpMZupfh9X0294NIEpxfcv4J7Irg0wQ'
                                                'UL7kfBHRJcXq6OtUt1zBBcnuCmCe6h4BoEd0RwwW81TJgg7rrg'
                                                '5grug+Csh+lYI7i7gvu1WUdJGz3nBeczSsNrwdnv1LBAcNME90'
                                                'BwZwUX8FS8V3B3BTfoax2dTotnCO61l/AYpXqChXZmCe664E4I'
                                                'zkOnxbFJy+eCmyU4lzyxV8GFCu6W4MaM1zNFcE71eg4f1+P6jw'
                                                'YH4WW0gssQ3AnBnRRcgeBWCq5GcE7WeioEZxJcguDeCU77Vk/z'
                                                'EAP7BbdacHbPxI+bjnWCuyW4I4JLEdw6we0Q3BzBeZzQ8z/BtX'
                                                'YVfRtnIM9sIE5wMYJrEdyAeh2tzuhYKLjbgnsiuMOCKxHcLcF5'
                                                'CU4uuIWCW7/ewNYzBv4fI/NcWQ=='
                                            ),
                                        },
                                    ],
                                },
                                {
                                    'array': '<f4',
                                    'shape': [4],
                                    'zlib64': 'eNpLS0uzP3vmjK2xsbH9rJkz7QA8uwb2',
                                },
                                {
                                    'array': '<f4',
                                    'shape': [10],
                                    'zlib64': (
                                        'eNo7IfXAjoHhgd2H+od2bKs47adxG9gzMBjYP+QwBNOsQNwhYGAPABmmC2'
                                        'c='
                                    ),
                                },
                            ],
                        },
                    },
                },
            },
            'x86_64': {
                'research': '7440581b1',
                'recorded': '2026-10-23',
                'jax': '0.6.1',
                'jaxlib': '0.6.1',
                'numpy': '1.26.4',
                'xla_flags': ['', '--xla_cpu_multi_thread_eigen=false'],
                'values': {
                    'adc7cbea87493ea1399fa2e2a0a398dce751c313a4959615483b5dc63bf15d31': {
                        'tests': [
                            (
                                'test_anti_node_rule.py::TestAgainstLiveAnti::'
                                'test_equal_on_antis_arrays'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 10,
                                        'substrate': {
                                            'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'max_depth': 3,
                                            'variance_threshold': 0.03,
                                            'neuromodulation': {
                                                'mode': 'true_neuromodulation_4nt_option_a_tanh',
                                                'modulation_strength': 5.0,
                                            },
                                        },
                                        'neat_species': {
                                            'compatibility_threshold': 2.5,
                                            'max_stagnation': 40,
                                        },
                                    },
                                },
                            },
                            {
                                'problem': '_TwoInputXOR',
                                'get_data': [
                                    [[0.0, 0.0], [0.0]],
                                    [[0.0, 1.0], [1.0]],
                                    [[1.0, 0.0], [1.0]],
                                    [[1.0, 1.0], [0.0]],
                                ],
                                'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                                'input_shape': [2],
                                'inputs': [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
                                'jitable': True,
                                'output_coords': [[0.0, 1.0]],
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                            },
                            42,
                            'xor',
                            '<f4[4, 2]',
                            '<f4[4, 1]',
                        ],
                        'value': {
                            'tuple': [
                                {
                                    'tuple': [
                                        {
                                            'array': '<f4',
                                            'shape': [10, 2, 340],
                                            'zlib64': (
                                                'eNrt23tQVNcdB3A1mPhoMZoERSPRKBJUJEZLfEz2/kLwiWiCKI'
                                                'IJKFrDVKyv0FQbLYhYggpGJPjYAIqw8lZhebP3ew8aRo0gHbI+'
                                                'qFZAjA6WiDwsaKY9caZJp4wzwD+mw+8z8xuWe/Z7z93LzuF3d+'
                                                '7O3ZlH+xYtJ0d9KC0c6oCcFhOtjMmlgreP0d2RbpTyUT7Z+Rko'
                                                '4qA/lW6ZoltRmEoJzkHUcmSJYmf/DubtC6d29yLlzuhXsXHtQf'
                                                'RijLEeauZFE/nUxVPIhRQyWDoq5cs1qlxvIn+rHFp05SuyHKNS'
                                                'tHUepZ5OoLerPcjtQj7NPXqCPvFbTx+6GXXXStNIPzmE/McEK1'
                                                'X6qbyeMsZ6rLrx7srLramwX7peveWTjY9y3WnwkEOq4yv7YeFl'
                                                'RODv7alydy9k9dUDw/PgVt+kBOocEfDPBDy/oABzCsMVY7onYh'
                                                'ek4OqAIl5PGWM9VkaFoLKJGgUUCTo4AfSnOkEueYI2jRA0qk2l'
                                                'hpuCgk8Iyr2lUex0lZyrBJkPCPKK18g8z0QOlYLmfSbo/nyNVo'
                                                '4sJj6jjLGealVDLH17z4ghFwvRcFvAajjIq9dxihqaoUaUZUEb'
                                                'cIheL7bGKlMuEr/TUGLljjOJBbByEHj3loDFhSKcXyGguyuwrl'
                                                'X81J/auxWQ13tJ5DrElUqmh6v/O++lyKNUvsuRPF6pUC87HPgp'
                                                '5z07l6KH6SljymA60ti/Q79rq0XTwhM3Ffz9DVisju8w3hR3WH'
                                                'kwaNZT+2TDGh80rU9+Zn30hr37qHnHB1ThlaH0eWGD2tX8hc0u'
                                                '9PG2cGWis16taZvf6dehrg2j0nQnGl3pqtzLVrs8r8OUN6jtcG'
                                                '8lbGqN6pe+qtPzDnPYThn9rMlOeOv+XN67y+c94zf9n/yP/q5+'
                                                'MBKjtnQ671bnT9FhjcrYd1TTi6tHd3lei5ByxTR1vLrM8U3c8N'
                                                '/V6XyL+X2avjVNeVS7Wb0+R9fleV+KClNqx8Wqs1JcceViBF/v'
                                                '/Z+JeyioXPagK8oExQw6RRXy90RZ9S2CmmsFRcvHJB/bXxcUWK'
                                                'vRimZBA2VPatypUegXWVRWKsizr0ZbxmRQq+tw3X/2e17mUmTN'
                                                'lWXZKjr0rQly+1uySuS+JzX+PH5GbtPLspa1saVjbr/c3i73N0'
                                                'iO3WnoOG6WY3HN4ql98gg5VlUvnlkfHS6P303WTXmcr7Z2/Tic'
                                                'ZbZM5h7L1x/S3Pn8dplzklUgs5dbuj7veJk9LrOnZfZxU+fzG2'
                                                'TORtYemd3VjXkHy+xmmfWW2ZVdmNdX5vrI8pJZ+27M2yhzM2W1'
                                                'yHOc9qDzedf/+tsWNXd93osy90ge7/Yf36eNz+59yrpnRtJphP'
                                                '4rDhFxC/DD2XHqX2cWwHdmDlKXn0RuvgH+AzKw1v04nEd8iT1x'
                                                'wei7Zx+8gzbDtWUqXrM9rwZY2sB2fJTad2G8zu61GwqfUcZYT3'
                                                'UuXUNfZ8DiTCHWGbLw0iyB6mYNB4M0XLsMWC3WcDYaiN2hYuSD'
                                                'YrlBxV2PYiw8UwD/53IxSOTDHGaEx5JMeE5L4usTxliP5fjpMe'
                                                'w9eRpXLQ1Ibcvm9ZAxxrpp3V2BrfcF1tTLn42C11PGGOumx9NG'
                                                '4/7Rgch019OcCfFkMyEU4l4wllhuQ8yDQMQX55g2nxpp+vG5B+'
                                                'YG6z6z30Sfrt9C308Oog93hlBgdToVN2fQpbZMWtd4kviMMsZ6'
                                                'Ks2cTAM90+hYDcjfSqOVW/ZT6PcHSJ0dQ95Bh8hvhpGy7xnpRn'
                                                'QOLXXKpVl9VGqYqlL/36qUv1+lYZc0On9DI+2eRi+0a7yeMsZ6'
                                                'rDGLj8HjcyOeK7HH8b9FwXZyIh5WZ8KYnYMPBhfC5r3duDQpDo'
                                                'sGpOE+ZWPK6zNhMz8Ik9pj4PM7A8Zum6fqVwyFbeUaBPpE8ucF'
                                                'jLEea+mRneivJGP686dMNxI8sLX9c/gti0VrUCoW3s1C+TUn+O'
                                                '7dhqC2L2FzNgnf3HlLjYwZgtQFq2CYGIEdwZef3CPVe9y3al6D'
                                                'C6+njLEea8YXnmpJaJRSXGOkc2MLqXlrALJnOONMyiD0+0OqGp'
                                                'lYobQeHkXKvvfpzUl/JEPAMYqsTqarsZn0db9s+tWIInqYVEyP'
                                                'rphInFU7XO+bm/yQXOWGSclOsFs9ssN6W3p7t+qWbv3k+zoNfy'
                                                'l/8n2Aqj5m3dOONznTgT5JnMWfKzDGfnFcwgRadQKjTufj15uy'
                                                'cO2cgKFIICVVoCxaYI2twK07GhoPa9jqpMHsqOLdoSb4phVhr7'
                                                'kAE3WncOSHVNTYJ8Jqkb7DevmPrwW8IVCXJ7DhZMf7ByhSwDFY'
                                                'YNpGgeG+Auc2CHzl8/T7DIRZw8VMjftgxtgvTqmiUeD1aDpqEB'
                                                'TxciH3fYwx1k15i4poWe1e5XaJRnXZCbyeMsZYN5nXJMEWs9Hk'
                                                'cgoXAnbxdTRjjHXT48UaTNuL8fF9Dd67VV5PGWOsm/4NllTK2Q'
                                                '=='
                                            ),
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [10, 340, 1],
                                            'zlib64': (
                                                'eNrt2H9MlVUYB3AUxQYhzDUH0VzaaJEk1yBoQfd9colMh6wRbJ'
                                                'opk3VNu44wx8IABRY6uiUkpnTRoLuLGr/uRQTuD+75vs9MqUQo'
                                                'YTBgWKC0aBoNBw2ldecf/iFzA/6pxvn8e3bOc97znvc5z3ltMU'
                                                'UonnwHdclautybR1OR3+B5iwkJ+4yIX16MrB2pII8YdN3yxZ6u'
                                                '8yLm+EIlbcCqpHv7Ut0r0ZTzWz71/VxEZzYY6c3PTOQhSZI0T6'
                                                '29dkOUXW6mc1f8yZ4t6MWNxzFqWKH0tlVQwuEWCrm5G61hz9CC'
                                                '1jq61CqoKNQLX5Rk05bOJuoPVO/nTyMZqTTTSRHXVZlPJUmatz'
                                                '7pZFT3mHFHJ7Cu4wDVjDEuORk+1QK/ZBdDDDLGflCxvsuKAp9G'
                                                '7W0DI+4vJ2xPJ6IQpeTXAzToTIh9YxWZVzse5NOjQwx9FePDbh'
                                                'V3rS48HDfse4ZlE8OsBzK0tgftJ0cZRf2Mx4oZX36gTutX+Kt7'
                                                'XDsjxJ/xQqGY1h58jXFoL+NRzxtpZkwKFf/WenslMESdioRMIP'
                                                'hJ16znkTKpYsBbRWiUwPYKx4z7LzrGWBDBWPWRilPLMOu4eSmM'
                                                'iz+qsDqAl8pmPu8yB2Moj1H5BMNfO/t133+KsSmOUfG1CvuOmc'
                                                '+7170P7lkZYe8z1o3OPm6P6t5DRxmaUMbv7828f7L7ewm6ymiv'
                                                'ZGRu5VnHLelmRNoY0TmMEh+GzFD/L150hJrTB7VZVhO2J9mhmb'
                                                'BQUmk5nbiQSxlrAmi84TUKWvKW4unfJh533/0nohVM7f8US4Yr'
                                                'Ua40oLTJgvptNuwabEHa65DvX5KkeWulRqDHfZ7eThG40SHPQ0'
                                                'mSpLk6cFpQVSjT6XcdpGhU2uAjyOinkqc3U38+05/LWuhuEMh2'
                                                'TKX1G5n0gXaqr3FR4iBo0R2VFp+8QFc/dtJ4tyCxT/4/lSRp/i'
                                                'rYFqD0+Q0rBYbNsNT6I+LeWcUzbkwZCXmKUsujKXgyVmicga60'
                                                'FePa1IsGpSR/Ney7vHF94IpYmHhIjJXlIndSj78NiTjbGSnrW0'
                                                'mS5q32jqb7NWXWWju1Je9UdGktFL7nHLXrVlJN5quIW+oiU1I1'
                                                'daovU+3BZxFudNGtkVqKjd9Mb3/ri6gAQX+EW8nhuZO29A2Lh8'
                                                'dvtKeQQVeoHC4cEUuP5EzLt6aJ3bR3zRklPaJd3NyagamDgkY/'
                                                'dzyyzl1+XtBXHk5ZB0uS9J+j7Wf6SWWKYjvFV9XLPCVJkjRH+u'
                                                'ZGfLcYiHmuEeYhIe/rkiRJc/QPKwHoPA=='
                                            ),
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [10, 340, 4],
                                            'zlib64': (
                                                'eNoU131czff7B/CDtBCiKElCulPpvlOdz3UdhsXcxDLNt1mzbM'
                                                '2vkblrFg4yIcRCCCEkIUnC+VzXQUiS3GRROBoWwxptmrXtd/XH'
                                                '85/zeJwePc77el/v18uQ5o5L09zZwW0dOLqto8beC0FQ0NIlpF'
                                                '26BAomBOGRCUHsff8UCBqx7RUI0pzYGyHOzjP64FyjD2f9sAe2'
                                                '/7CHei7siYKjXPvAWNc+pIVCEHR7fXeqXd8dDLbXjULnNr6BPM'
                                                'Y3wNE5gSg4+WExLHpYTBnL++PG5f05LuUriE/5iootn4Igl5up'
                                                'Rs3NVF1tkQZuF2lojLmExppLwGJ7KHbYHsreUy6Ax5QLNMnWEw'
                                                'UnuG+Cme6b6Okf76FgQ+hZRaj6nmkgKD50BiWEzoCYveEoOLVP'
                                                'FSzvU0WN1n4ouHjsISgZe4j0RnsUbPDVgqDkhRdhwcKLlGEgVS'
                                                'gWo/1QsIVDPrR3yKeoM24o2HrdGhCUOdIBfxzpwE3TI6B5egTN'
                                                'MrZHwZrA5UpLwHI1Lf8y/JB/maIiilWhlI+fCYI0P65sA/rq3o'
                                                'pQl6+8T6kr74M5y5UE1B87QAKM/wTg2X8CuDnqBAgqqvFCwVnP'
                                                'tsL2Z1tpfLIzCjZZfw5G68/JprELCnbR/aIINWnLIxCk/+i+Ue'
                                                'hSbm0BQTZbR5HV1lHgYniiOBieqPEWKgkwuRWpQtE5lZOAhrHB'
                                                'KNg85iw8GHOWnKf4ouDKtfugYu0+8rg8CD0vD+ICl2UgKPWOLa'
                                                '68Y8tRbgNgrNsASl7cAoJMnn/ohNGl+CQ4F58k/ZQObaA6QmlD'
                                                'hbCDCmAHmPK0RkOeVmfs3EgCUkeGomAb3/PQ0/c8maP9UfCL3C'
                                                'PwPPcIXYryQMHlRzdA5dENFJTviMH5juxSMRpcK0ZT9vuWuON9'
                                                'S25am6cINSPjBgiy+WKlKpTWu8tAUMbdOW3AkJqqCLUn3yA7vg'
                                                'HJzVpc2Kxlk3M5kHM5jW0IxA8bZJ6/KIEFX5RQoZ83Hvfz5oLp'
                                                '2ZA/PZuiVrrgxJUubL74DQjK+LMbbvizG9c2W4CgxtjnIEjTeP'
                                                'KsiEhO2wsL0vaSX7YPeWT7QF56NyhI70bhRYWkKyqEvPhwPBgf'
                                                'zq26a/BOd41au4fg391DOPq1CkLu8FAMXTqU02cdBEFeaYNRcE'
                                                '7JDyDocXJvFOxw2A/sDvuRi4MGBzhoWHPtQ6Wp8kPVHGUCQTGx'
                                                'd9S42DtKU/EUEFS0dz0V7l0Pcbsj8LPdEWy3/wb03H+DHF+Eom'
                                                'B9chkISpsYgKsmBrC5/XF40P44RVZ7ouDIjEwQ5P3SCQVnbJgC'
                                                'mRumUFN2JxSck35NEarGug4E5fQYqgolLXA9CPLeEUMeO2LAUK'
                                                'BDwdWza6Bqdg1dWh6Ggl/YV0CTfQU9Px2Egs1jT8ODsafJItkH'
                                                'BY/dmwOCus0fiIJLG78DQWV1PVBw+a5eULarF1kZX4MgQ8j3Om'
                                                'FMrD0CgjLNdpRhtoPmJH/8I8mfDRZHYYnFUUr4xgcFNy3JgRdL'
                                                'cijhew8UHBSWAYJa4gfiX/EDOWnFdyAo9bkjrnjuyKYJH0LZhA'
                                                '+p8EIPPH6hB+tte4GgvL2WeHCvJRfuKlBKdxWofgteQ8AC+X/+'
                                                'mK3T/DnbmJN9E/Zm36SmaQZVKMnfHQFBmUW9KKuoF9QHrWhDBq'
                                                'fZbcDKwrMNFR3MpeKDuVB4dI0iVPO8ano8rxpyAixUoUR9/ZAE'
                                                'ZAy0JAG184tJgM75S9I6fwmRe1bRiD2rYNH5ABRcsLIIBI3pOh'
                                                'QFO/y2HwQZv/dC9Xsvtjm/BQTF17mi4Pr1qSBoTFM/FJx06VOY'
                                                'd+lTCprfC4Pn92KPw14gqLxfF7zSrwsbpj5QkqY+UK20/4Agl7'
                                                'HeilCTox+CoCRNR9Wg6agkBBtBUK3/M9Xs/0ypvfMjCMq7P7YN'
                                                '6BPGgS5hHBVkymeZP4Jfyk+KUKf1NFFcTxNoFqoR4mz7kc/JYu'
                                                'Rz8BtpUoUyvPdFGtH7IsQvGE4JC4ZD1eUsqr6cBSPeBaJgj5IS'
                                                'ELRgnR8KLvsvHy79l092k73RdrI3a+uyIaQumzwGuqP7QHfOq1'
                                                'sLuXVrKcHogoKz+s2G7H6z6d0jBxTsYI0gqDqyOwqOvGoJUVct'
                                                'KXtvBxQc9WytIlSLMb+CIM2RyWc1RydHWPlWQGffCsqblacWzs'
                                                'pTnC7vA0FNz73agEdpUhsa//VyGvv1coha3aMNjZl6jMZOPQZx'
                                                '3wQqSd8EqhXb7lHltntgiP1SFUrm1BoSEHmpHwnIqzpEAqIcg3'
                                                'GCYzAH7TgDATvOUHSTP37U5M/FJcdAUExnX/yksy+b1+yFhjV7'
                                                'qVTniad1nhx/dBNMP7qJKicNwquTBnFc3FKIjVtK47s5oWCTMg'
                                                'nKlEmk8bBFwTle/SDbqx8ZJlmh4IxQVoQatexPEKQxXdW5mK4a'
                                                'Gz//CZ5+/hMZ1EmqUJJai0BQVFIniknqBH4JqyEgYTXN+346JX'
                                                '0/HVp+CGlDdl/sagM5PQ4qQvVNriDv5Aown9psNJ3arMv1fUJ5'
                                                'vk8g7dsm1fBtk/Ju5xlq3XkGklxDcLZrCBvDCAQN1Aai4PQ+xb'
                                                'C+TzGN2DUU3981lIN+OgghPx2kWN8h+D/fIZzQdRvEd91G+lOD'
                                                'cdipwRw7bhVMHbeKpv/rjILjRseDoFLujae4N0dNCABBeces8e'
                                                'Axa256r0nRWDWpTUs1+NtSDevnRCt+c6LV2hs/gyDzy0Kj6WWh'
                                                'rjLpHAjK8L6hCsWqYhtYVmyjmI/0baBl9dQ2NHbEujZQnfxGEW'
                                                'rem9MkQO++V+fivtc4fcNjit/wGDL+26cKpdn5KgmIcQzFKY6h'
                                                'nLLlHCRvOUfNXwah4AKfUjjqU0pxtv74ma0/e18qAN9LBWT3zh'
                                                't7vfNmzX+74N9/d1HsS3f89KU7pySuB0GjSgag4KSw+ZAYNp9G'
                                                '2DniSDtHzsNRIGgE2eD7ZMNJ3t1BkNeXHVFwda/dilArDb+BIE'
                                                '3piwjNqRdnbQZcB0Ha9htVv/YbldgFh0DQ41XOZF7lDE0zF7ch'
                                                'O+9kcvBOhrRh/dtQ3oBDbcBlbaLitzZRXVBym5JLbkNUuI8aF+'
                                                '6jhOjqKEhXBxn/hOLGf0K59mQZ3DlZRqVLgvH0kmD2azkLAS1n'
                                                'yTIxAAWnfnocBJXt9UXB8atzYfrqXDK0eOLSFk82224GQfVhri'
                                                'jYmL4cBDV59sPfPPuxucMn0NjhE8pZbod7lttxjP1giLYfTC5v'
                                                'O6Hgpu9uKEKNn/sOBJlduivVLt3VTOt6EKR5f7AqFO2MUgiZUU'
                                                'rlq96p1aveKYmpG0BQuXlyG6j95f025Ngni5z6ZEHSRxcUodrM'
                                                'KiMBmuNrz4qIqIbnFN3wHLZc16Jg6+zL0C37MmmXhWDYshBuXc'
                                                'zw72Km+PxAnJEfyGV7T4IgmyF+2H2IH+cVHQJBt+uGYE3dEDaP'
                                                '3gENo3fQvXQ3FFz+bjVcereaHli4oGDz/plQu38mubRzQMGlV8'
                                                'Kg+EoYmUO64cOQbhx3TAOCyqe0x8tT2rP+1feK36vvVevgRhBk'
                                                '0AQYhS574yUQlBV7Ws2JPa1k5O4GQbk1gZRTEwgxqV+1oarff6'
                                                'Dq338Azb73wGLfe5RUcaINaBI6KzYJnVW/8gfkXf4AijaGoWCL'
                                                'cRXQflwF2SWFom1SKOdcOA97L5ynH6qDUMh+PA3aHacpcbM/fr'
                                                'NZ9qPhKJQajlLMdh+cst2Hy+32wCW7PWQq8sBzRR7s8DIDer/M'
                                                'IJtDA7H7oYGc4Pc9CLKCvtgJ+nLamHEg6IBnTxScNt0e0qfbk7'
                                                'fzeyjY8OK4kvziuGqz6w0I0tRu0Jl/2mBM+OsWfPnXLco5OFcV'
                                                'SmOXYyBIH9qTIkN7QnNLahu5G//XBpLrvNtQ7qO9lPNoL1QbNy'
                                                'lCbV9WRRZlVeA6MhwHjQznhMpKmFlZSZVjtXhtrJaTr12ERdfk'
                                                'zTQHo+DGD1V4+qFKHo8D0PNxAAf9VQSCqkKGomDjIukyiw6Q+a'
                                                'gXCtbM3AoWM7dSjONg2S+DOW/KSsifspKyFWcUXLj3MxBUeasX'
                                                'Xr3Vi0vvekPxXW8yZXTBcxldOGfGz4pQjWP/BUEaNVSxUUPV6m'
                                                'wzCNIfaDIKXc4JFXadkJw8rUHVT2tQXqRtBkG140e3gULXiVDg'
                                                'OpGaKza2gepZDxWhbppClDmFwOrXcBRc2u86nOl3nW46hKHgvN'
                                                '3lICimLgSn1IVw+mkTrDltovg+QfhFnyDOGnAKBFWd8EPBpmWH'
                                                'QVBUqjdOTPXm2oU74d7CnZK73XFKrDsnbU0HQU0dBuBvHQZwcn'
                                                '0SCMoN64P7w/pwjuMwyHMcRnknuuOhE905eWgnmDe0E9EgCxRc'
                                                'HbdRqY3bqEYteQHjl7wgTcuG0xpNRljZ1qsgKOniHlUoVl/uB0'
                                                'HFX7pT4ZfukDN5XhtK2mVoA0mze8G82b3oztYjVLtVsnBqBH6T'
                                                'GsF+I26AoPiaMJxRE8bNylV4rVyl2Auh+L8LoVzceAGKGi+QTW'
                                                'gwdg8NZuv/zoAgsg1AwXlDC+HA0EJqDfbFf4J9+bHTPmhw2kcp'
                                                '33ri4m89ufLTH0FQc+YgfJM5iDOWGWD9MgMt+cwJBXv4RIOggL'
                                                'W2KLj8m/4gqLrCCm9UWHFTYpliTixTrfe+BUH64T/rNMN/Njqc'
                                                'qgXHU7VkqhypCiVlVTEIannXkTR/d4TqwWuhavBaim+cRgmN06'
                                                'AlOrwNTd+RTXE7smFETx2O7Knj28NvgaDSFeEouPXwNdAUXKNL'
                                                '27QoOPP+Jdhy/xJpMAQFp24hWL6FqDYmEAVbrymGzmuKKfPSUB'
                                                'Qc9FkeCLKMG4KCnY5vA8fj2yiheTB+3TyYx/ZIA0E/jO6PgvMc'
                                                'v5Tz/5KK+9rjib72rPk2CFrnBFFm/64o2PCuWUl716zWW7bDOs'
                                                't2bC76TDEVfaaW9XwCguKebDUanmzVGX45D4LyvqpQC7+qUDzm'
                                                '7wBBjy/ryHxZB0kR09rQve/WUu13a2HeZh0KnudUA4IaXSPwF1'
                                                'eZh5fXwffldfrzszAUrM2+AiHZV6ghIBQF5704B7kvzpHvyiAU'
                                                'rN9SCoKytf4o2DTiCJwfcYRa3H1QcO7W3bB3625qGOyBgnMGbY'
                                                'C9g+Td+nsAXv57AFe/XAC3Xy4gw9eOuPRrR84aHgmCdG49UHHr'
                                                'wUmpPUBQ+8cdUbDN/FzFan6uevtIEwjSbHXTCaN2RTWErKimqJ'
                                                'lrVP3MNYp1t8MgKNeqL+VZ9QW7JYY2ZNoyrw1UdlPwajeFS2/d'
                                                'gZJbd6jpdAT+djqCcy1uQo7FTfKwD0dP+3CO8agEQcPttCi4vu'
                                                'NFEHRgezAKjv/ECNM/MVLi+gAUHHnrOAhyu+aLgrPt94OgUm8v'
                                                'POXtxY20GZ7TZnJZ5IoDFrmy3ScroPcnK+hecj8U7NI4FQS5/G'
                                                'yHgg3j3SF1vDtFJXbGCYmduXDLHUWoduv+BkFxJxwVoUZOvQ+C'
                                                'mpIdVM13Dkp142kQZJP/RhVKbMRGENQ6QbJ51CSwWqqgYJv/qw'
                                                'VB59/XoeDoP26BoDRTOK4yhbNhZRUISn+sxXWPtRz99DLEPL1M'
                                                'ljtCUPCIWobIWibvikD0qQjksTYlMMamhEIm+aFgl9B8cAqVPm'
                                                'Utfcram5Oe7YBvn+0gfbkbDit346bTa+DV6TX0eJwL/jzOheMS'
                                                'E0HQpqkOKDjLHAGCbu7rhoLjnDqAoKi69jixrj1rDCsUK8MKNW'
                                                'vWMxCk2aYxCl2ZphwEJR0sUoWSF7kHBDmN9yOX8X6QdMIaZ5+w'
                                                '5sKAJsUU0KT6bfBDwdkdD8OujocpGt1QcF671VDQbjWFBIShYO'
                                                'Nf5SAoBR6BoJxBj42mQY9102J7omCb1Q7gsNqBWvLd8G2+fP9/'
                                                'a0BQ3uIAPLg4gM3bjoOgx4mW+HOiJedE5StZUflq9o7+uGNHfy'
                                                '63T4BK+wR6PMMXn8zw5bHqPhiv7iPtW8lnb0N4bN9zIEh/vTcO'
                                                'u96bsw4HgCCnUZ4oWHNtE7RWbiLzf4H48L9ATr1SAsuvlFBSSD'
                                                'jODgnnF1sroXFrJTneH4iC85JToCA5heaN9sO5o/14Xp98WNAn'
                                                'n8Z316LghOYyiG8uo9xNOhQ8tqd0yZ41ZHxiBEF666dqlPVTpf'
                                                'rkMxBkKHl+VlPyPGJavSUKLnQ4ppgcjqnr/7JDwYabMs833cnm'
                                                'QX8UnFX8NQj6ONADBY+YsAGGT9hAi9J9UXAu5MJeyKXquEC8ER'
                                                'fIyfuLYcH+YjLPqAVB+oZIVSgtv/0H7377jzQDRitml9Hqne3d'
                                                'UbDLHivw2GNFj4198YmxL5ePi4KycVEyj4NlHgezk9dKcPFaSc'
                                                '2DvfGPwd7sgtkgaNo6fxSsm3EU9DOOUsWhYBTssN4IjuuNlGxq'
                                                'ggUm2T+Znjph3JXbCQW7NFxTHBquqU8H2KNg85YgeLwliIpWDk'
                                                'DB5Vaye6zmUcICT/x6gSdnzfwRBEUtH4qCW5MPwr/JB+l23yAU'
                                                'bOV1CgQ17QjF33eEctOnF+D1pxeosnMHFBxlWKkI1aZdT+zRTu'
                                                'ZvYW8QdH5zPxRcuO1/IGjqfDcUrD22GkKOrSarMB8UPK96N3xb'
                                                'vZuypwXgjmkBnD30OAiyGh2C740O4bTTBIKmjghDwa7hV2Bg+B'
                                                'WyG2+NtuOtWXPmV6Xl9K9qN/8+KNjqFz0ISu4zCL/rM4gb/1gM'
                                                'gjzyvdAz34sNU7eCoHmJMm+JfvxiQz682pBPzdeC8I9rQRy16T'
                                                'RM2nSatgySfjJIy80RF0GQ3dFwFJzyfhUIKv/QDgUn1QyCxJpB'
                                                'lB/THwUbkr8EQenX3HHdNXdO7rgeBFmgL3ZEX3aJ2AeCHg8OxJ'
                                                '8HB7L53glouHeCSs0heMocwnGVJhDUdCoMBdvUVEDPmgpKb4hA'
                                                'wfVbbsK9LTfJyqMvCm4sGwsvysbS8O2uKDiWV8A0XkGR14fg6O'
                                                'tDONFlB8x02UFJH/rj7A/9uTz1CFSkHqGchcG4e2Ewpzw8C0se'
                                                'nqXqXC1ez9VytsVl2GVxmXJtInCfTQQbcq+DoNJnOhQcP+4OCE'
                                                'qzHICrLQew6UUSGF8k0bxenig41ncTTPPdRGu0Q1FwpuEAbDIc'
                                                'oJKfA1FwdEIJfJxQQto5oRg6J5RzL58HQXZu4djLLZyzPq2ELZ'
                                                '9WUnaIDneG6PjFxVvw/OItiqlQUHBzyl14nXKX8jblQf6mPMpc'
                                                '5EJZi1yg+Ndzbai6uUo1NVcpt/+tBUE2tu+rGtv3laRrL2HWtZ'
                                                'ekiX4/QjP5/bPaYA2GBmvYoB2vJGnHq3mLrPDQIivWN5sUoRZX'
                                                'dMeTFd258EQnENQ+ojcKjtMNBUGXHvVFwXG+E0FQ4W0XFKw9OR'
                                                'v0J2fTCI/B+L7HYHZJXQnOqSsp6TcPFJxetBEEJUZ4o+CgH7Ih'
                                                '5Ids2tV3KAoe0f8AjOp/gIp2+qNgc/ZREFR8LRAFJzmVSF4qIf'
                                                '3lUyAobtI/atKkfxRDhZxTxXWKrFun6uvWKdofnoAgw8P1RqEz'
                                                'Wf4FgvT+GkWoPwR2RMFRL7MVod57Yo2CNd+8VoSacMwWBScPc4'
                                                'GUYS4UdaAPTjjQh9Py3gdB683OKLg64wsQFDR3EAou7W2AM70N'
                                                '9HiOOwq2uZ8O3e6nU9orLxQcGZEFYyKyaPhfPii4bOJeME3cS8'
                                                'k5frgwx4899IfBS3+YbjcFoOAC1xPSo05Q2h9BuOqPIM7oewbW'
                                                '9z1DiYGXQJDe7qwqlIQH9SAoLqyfKhS7D34HQZoA1Amj4Vw7FK'
                                                'yZOFtxmThbLbzbCQXn+FcrQtVpJO9perDVIRuwOGRDmg/tUXDy'
                                                '2mAQ5FbthILN9z8GQQH7B6DgqKp5IGjmg8EouLxiFVyqWCV5zB'
                                                'O/We/JVqd/BEEul7yx/yVvjpy0C0ZN2kVbMoei4KySgyAo4L0A'
                                                'FFxtUQhVFoV0xjMIBVtPPQWCDJYhuMwyhCNvqDDqhkrlt2+BIP'
                                                'N781XTe/OVSM9nIMi0qovRsKqLzjCqFZaPaqWcyQMV0+SBammu'
                                                'JZ7OteS4kwVKwskCtWVINxQcOUcDY+doKKbKDqdU2XHkbTcQZL'
                                                'XXEd/b68g5WaNB0FNjfxSc/DQBUp4mUHmgK14JdOWk75bDrO+W'
                                                'k5WjBwrW/rMeQv5ZT7plQzBi2RDOdd4OgsK/80XBdv/uA0E2Nv'
                                                '4o+MWZAnh1poC8JwXikEmBbDe3GAQljQnGOWOCOWH4Wfhy+FlK'
                                                'bB+KgtOHnwNBGYfNsOmwmVyW/2bUL/9NZ3WsGTofaybNr/t1wr'
                                                'hoZAcUbPNulSJUa/8u2MW/C2cEPFSEWunbE6/69uQEgz0IKo12'
                                                'wNPRkvdUyXtqBL0q7YeCzXNioX5OLJUlDUTBhde+A0EhGW4oOK'
                                                'PjGtjUcQ1leXrhNk8vLijeDEeLN1PKOB/8fpwPp9nlwBq7HMrt'
                                                '7IeCbUYegm4jD5HHrAAUfDv+OAhasyIIBWdllEJmRimZo0PwYX'
                                                'QINzYQPG8gev48FAXbzSsDQWPPvABBGr89YeK06/z/QJDJcoQi'
                                                '1OauVvhHVyuOG2ZUhFo9szsKbln3HjSve49Kn/bCU097ccZoHx'
                                                'Bk3toXH23ty2mJEyA1cQJZZbpgp0wXzhswCwSV/+aKl39z5frr'
                                                'qfDgeiq5kgcOIg926rkRBM208UbBlW93QNXbHeT4py8K9t6zH7'
                                                'z27Kd3Kf4oONHnKAjyPi7nfTyQHc6chN5nTlLUzmCcuDOYC6cb'
                                                '4eh0IxnHhKLgBP15EJR6Tosrz2l5xMzLICixYwsIMsX+qhNG7T'
                                                'kLDDtnwS42WYqDTZZ6x2CNgm3evlSs3r5UUz62xcUf23JjmDO8'
                                                'CHOmyJg+KNh8cxjU3hxGumxnVLKdOef555D9/HMyhA7CpaGDZP'
                                                '8vkf2/hEyKO55T3Lk4KB2KgtLJ6pwXdjon571/K+Tv30qZN3xQ'
                                                'sPe1PeBxbQ/dW+yHgs0V+SAo92YACnZSi0BQQF0QCtaZTkO46T'
                                                'TF7AlBwYmvGGa+Yirw0aLgtM8uwprPLlLuqjDcvyqM9Z4VMNyz'
                                                'gmaNboeC496foQg1dnYn/HR2J/bbeFXx2HhVNey1wWV7bTgpvR'
                                                'vMS+9GuS29cX9Lb87YEwiZewKpdZUT/r3KiQ2uk0GQ3ZcDsNeX'
                                                'A7h0+FwoHD6XRh0cjIK9HVeBoLHTPHHcNE9u7PsjPO/7I7lme+'
                                                'PgbG9+fGknPL20k1pmD8W/Zg/loIiDEBBxkEb84o/v/yJ5Y+sx'
                                                'qNh6jMxdg1Bw/V8lcO+vEgp6FYzBr4LZkKKCoOiNofjRxlDO0F'
                                                '4AQa7/aXHwf1rOGlEO20eU03oIR8FNxZUgyBxkiQ+DLNnw/KAi'
                                                '1IArXVGwzdJWRajzFtrh/IV2bKpxBWONK1VPdsQbk2WfXf0Acq'
                                                '9+QAGL+qPgpMivIDHyKzL8Jef91yCONi0DQa6/uKPgqJj1MD5m'
                                                'Pd2OHoI10UM4KH0bCFoyyRcFR6fsg6iUfRT5yg8FB0UWQEBkAR'
                                                'UHyXsdFMiVXYuhvGsxkV8wCnZ4dgYEtf4agv/8GsKP35jg6RsT'
                                                '3ZylRcGFIy/B0ZGXpP+G4WpTGBe8qQBBC34JR8EvbK6DIOsrnb'
                                                'HLlc6ckXtPScu9p3rf6oFC3u9e0JrYi0ZYOKBgQ30YpNSH0YJv'
                                                '+6Fgs2kq1JumkmvgQBRsHp0MguKnuqHgrJjVsCVmNeW888Q97z'
                                                'y53nEz3HPcTMtdfVCwS/puENT4dCg+ezqUYyvzYGplHv07PgAF'
                                                'B70rBEGxCUEoODWgFH4IKCVXJQQHKyFslUEgSHsnFAW7DimDgU'
                                                'Nkn0WE4YcRYdza5woIspsreXtuOKcnXYO0pGs0a0UECs7R3wBB'
                                                '+s7dETt354QZHSF+Rke6ubEXCraKHgIW0UNo3ri+OHdcX3bJGw'
                                                'dOeeOocZILPpvkwoVrE0GQucQVBRscU2GJYyolp3vgwnQPzk3K'
                                                'gJykDLJ7MgQFJyXsgG8TdtD4al8UrB24H0IG7ieLWH/sEOvPfu'
                                                'oREJSQGYhfZway9ZST0G3KSdKlBqOSGsxxXYwgqDYoFO8Gyfv1'
                                                '5hyseXOOEo9q8f+OajnHTfK222WyaQnDHi1hnHL6KiSfvkp/2k'
                                                'eg4HK6DoIMtjpcaqtj3Qe3QFBza08UnDXUCQQVdeqDgtPeIaS/'
                                                'Q5o1xRkFNw6Lg8fD4ijm74EoOCt0MWSGLiZtJ3cUrF+7FmDtWg'
                                                'r40QsF23XfCoKyDvngtkM+XPDBHhD0bqofCq6Mzofy6HzSFQdg'
                                                'RLG8X9FFIKjAFISHTUE8Iv40CPJbE4L+a6TP7GQQ0se0KFjncB'
                                                'HA4SK1XxCGgtOfXwFBsQfC8dMD4Wx2r4IG9yrSFUagYOOjG3Dm'
                                                '0Q3KyNLhhiwdF7jXQL57DZXs742CW4r9obXYn8aiEwrOSfgIsh'
                                                'I+IoPbABScVzwHCovnkHH2YDw7ezAnJ/wAixJ+oNs+nii4evEm'
                                                'qFq8ierneGPdHG/ODt8JWeE7ye7DodjrQ+nDdADe0QFyuOyPgi'
                                                '28joGl1zGq/j0Qb/weyJptJSCotCYYBbu6qyCo+ftQFJzw8jzM'
                                                'fHmeDC+0uPSFllPal8OS9uU0wjccR/qGc/miSqhcVElRH0TgxA'
                                                '8iuCChGo4mVNMDnQ4FJ9ffkvt7i1JsFVxsq7Dx8R0Q1K2TIwpu'
                                                '+XskCNKE9sd2of258eAMEHSJBqFgO8tl0NNyGT044Y6CY4vWgS'
                                                'AHjyHYx2MIV7ZkgaBFQ3xRcOlfe6Hkr72kr/TDYZV+XHz7MBTd'
                                                'PkxW9oH4nn0gVx88AYJqewfj3d7B7LfhDAiKrwnBGTUhXJxrAk'
                                                'FO/9Ni3/9pufW/i/Dvfxcp6lgYCo4rqQBBM2vDUbB1XRUI+rNO'
                                                '5r0ugl2X3gRB5aTDKyT973gNNB2vIZuVCgpuNdTCO0MtJQ/ph9'
                                                '8N6ccxXT8BQT3/GICCSy8sgMILC8jX1Q0Fp+SnwZL8NIq66omC'
                                                'rQyZ0NmQSbq/vVH5W/JK+91Q1X43PT8/FAXfnpQHgrYPDUDBte'
                                                'sL4d76QmoZG4R/jQ3i27dOwZ1bp0g/OAQFZ+kItuiIpp0LRcEF'
                                                'jy7A0UcXyM4zDHt5hnHs3XKYdrecfL8IR8HFodegNPQarZkfgY'
                                                'KLu92Aom43aMQMHQqOnnUbJs26TZUhCl4LUbg6/icQpHmlYLtX'
                                                'Cpf/excq/r1L43u4oGD9m5kgKH2xKwpu+WIFCEqc6oGCHao2gG'
                                                'PVBko8PQT/7/QQ9qjdDl6120mb74th+b4cvzcXBFWG++PVcH8O'
                                                'GnsEBH27MBAFR70pBkF/zgxGwcknzsKCE2ep0jkUrzqHssuBc+'
                                                'B84Bx579CiYEPDJVjeIL3slzAc9ksYZy6+CoJGdYpAwU5rroPz'
                                                'muuUZ6nDQ5Y6bnG4BYKcHutQsHXYHRBUuF/B4/sVzn5bC9vf1l'
                                                'LreEDBcXF1MD2ujoq/C8ET34Vw4RcMBV8wlSXo8EKCjpsW3QZB'
                                                'QTPLQFDMjAtq1IwLStSo7jhxVHfWqJZgpVqSUe6fKvdvXkolCH'
                                                'og+0VwpUcNVHjUUNkUwItTgBMNdSBo1g5EwU4DzSAoJcoLv4/y'
                                                'Yo+gLSCoeYg/vhniz6V9jkBxnyO03DEEBdvZEwhKOxiGqw7Ke7'
                                                '69AvK3V5DTgb9BkOZsX0WosWU2KDhnQnfImtCdTMXOKNhm+XSw'
                                                'Xj6diq97oGD9hxtBkCb8siJUy60XyGrrBUhY9H0bSuq3qA149+'
                                                'c2lHevTs25V6d4pL0Bt7Q3pLFdr3OxXW/MyIqQfRrBmek3YFP6'
                                                'DYov1OGMQh0/XlwDDYtrKHaXgv/bpXBjfS08r6+lygRAwdlUB9'
                                                'upjux6IfbqhdwU9ABeBz2gsoOIFw8iJ08wg6BiVz2edNWzoW8D'
                                                'pPRtINcNehQcm/szxOT+TJAXgoLTLU2QYWmiBUu0KLg15RK0pF'
                                                'yio13DUXCaZSWssaykirgIFOxRVg2CWg06/Meg45Sbt2HRzdtk'
                                                'tUDBTgsUjh9TC9PH1FLKKMDFo4Bjh9XBtGF1ZH4L+OgtcNOV+y'
                                                'CoNdoX/4725bzUfSAoa5Y/Ck5qPQKzWo9QVvcgFFz+XwlU/FdC'
                                                '88aEoOB0IlhDRPW+WhRs/OIiCLKrCEPBeb2ugqBY9wj81D2CLf'
                                                '68DoK8o3Qo2HXgbXAbeJuKP3DBkx+4cOPIRHg8MpH+XDYYBZft'
                                                '/QEu7f2BEl974jevPVmj2QztNZspu8QHd5b4sOvcPSBobKMfjm'
                                                'uUfhFYAA2Bkjd3St7cKflj4UkQVPssGH96FsyR36gg6F59KApO'
                                                'HV4GgjJqrXBDrfTDjhcVoSYu7YGCNWm2ICjumQMKNmgRBHVW+q'
                                                'PgqGszQNCk3a4oOOX6Clh0fQXZBHhi9wBPzlq7Cbav3UTzbH1w'
                                                'vq0Px360GwQ1feqHgqun5UPVtHyK7HgZBOn1p1ShGN8+AkGa8h'
                                                'qjvrxGF5/3FwjKqbJUMqos1fCB76Fglz5FilALLG1QcEuXLtDa'
                                                'pQt5HbRHwYWFoSBonpczzvdy5kLvz6DA+zMynxyED08O4vpnBh'
                                                'Bkk6IFQfXJO6k2eaeckaEN6ffPbQPNv+1tQ2lG7zbgXX6uDTVl'
                                                'XVeFoo++D8Oj75NfsYPqUuygJG37AwSZnpfqhLF9XEcUnHFzl5'
                                                'J1c5c6YkU3HLmiG5dObw+CGn9ZrTb9slpJ3VVNAjQvFZ3phWIc'
                                                '/uwX0j/7BQyVpYpQkwsvUUrhJcg54gXZR7wo8ed9bSDmn3ltqP'
                                                'eHBnL40ACadjvbkENKWBvQzjZC+GwjVXs2qkLx2FwLgqp7f6Ca'
                                                'en+glIXo8GKIjuddugULLt0iK6MO3zPquPVgDbw7WEMv5P35Vd'
                                                '4f7YyfQDfjJ9LnKTgsT/aB5i481dylchvAKzbAqYvvgSDtHMDQ'
                                                'OcAx9+tg6v068q4AFJw18D5sH3ifhjshCtZOeADhEx5Q61eIgh'
                                                '1mPARBDUcQBTfNMIOguFeIn79CjvzoEYz56BE1eupRcJ5vA+T7'
                                                'NlB9nB7r4vQc/1cDTP+rgTb9qEfBpcU/w5ninyn+vB5nnNdz/e'
                                                'ePQVDOSz3ufqnn2nZPQJAdhmMvDOfqkkqoKqmkgl/D8civ4Wzl'
                                                'ch0Ela2JQMGa/92Af6fekJ2qwzn9dOzwf7dA0NFcHQo2fFQDSz'
                                                '6qobgBCgp2sP8JHO1/ovhMBb/IVLigtBYEJf+r4MJ/Jb+F34Mz'
                                                '4fdofByg4JzddbBrdx1lngEUnPasHtY8q6f6roiCUxwfgKCSqY'
                                                'iCY4MewtSgh5SzB1GwWWsGQT1/RhSc5/kIBCX316Ngb6sGEKSd'
                                                'osewKXq+XdMAd2oayK8oFIcWhXJQwQUQ1LpRi/9s1LL1+UvQ+f'
                                                'wl0saFoWDXnVdA0CjncBRcPKwSSoZVUuK1cPzmWjgXnqiCohNV'
                                                'ZJ4VgY9mRXDln9VQ8Wc1GTvoUO2g48Rut0DQu3QdCq785zYISu'
                                                '2i4MouCtdfvgP3Lt8hnUHBCIPCibNrYebsWqr/VcH7v0re+/Mu'
                                                '/PvnXfKYAChY83kdCDLlA57LB847Vg8HjtXTiP8ABbc+vg+CdO'
                                                'MQBT9u/xCetn9IAZsRBXt0MYNXFzNVTw9CwTH9S2Fq/1JKWBmM'
                                                'X60M5sTuRhDUbWMICk44yfDlSaacVaG4e1Uoz+t9AQSVz9Si4P'
                                                'jQSyCoty4MBVc7XYEqpyvU0hqGb1vDuLniKggyHg3Hs0fD+fbI'
                                                'KhCUNSkCt06KYO+N1eC1sZrynkXgwWcRnFh4EwS9mKdDweYDt+'
                                                'HBgdv08R86FJy88A4sWniHyv5PwYv/J/M2uBZ6D64lzV0F/6tV'
                                                'eETRXYgsuksmRX4fBThlQB0sGSB5aTvgie3y/s2rB0GNvfzwWS'
                                                '8/9v6/QyAofYM/rt3gz3mJRyE/8SiN+TUABRf2OAEFPU5Q+pAg'
                                                'FDw27hSMjztFhknBuHSS9NtpZ0GQ1/QQFBzfg0FQYmwoCs5Ych'
                                                '42LTlP74ZrUXB5/kUQNNM+DIXkoXLJQ+XkXReGPnVhnDbpKgja'
                                                'siEcBSfcuQYz71wjc3AEPgyOYDuvaujpVU3e1yNQsMPEm9B74k'
                                                '1KjdXhilgdF4+/DUXjb9PtBzoU3Dz4DrwefIcspijYYYrCxXd/'
                                                'gpK7P1HrEi/8Z4kXGyu3gKDqr73xxtfe3Nx/Jwi64+OLgrPb7Y'
                                                'Pt7faRXZX0waqhnJacB2uS8yh+nD/OGOfP89YeAUFxRwNQcJm+'
                                                'CM7ri2jem0Cc+yaQc3aVwN5dJXTTORgF1+efgQf5Z2hTQAgK2S'
                                                'cEjv9HlOkXioIzHp+D9Y/PUZajFrc6atnP6SIEOF0kv9daFFzQ'
                                                'uxwEPSgNQ8FxtdKnaiuoZXY4Ck6YcQ2+nHGNjA4RKDj93HUQFF'
                                                'UcgROKIzi+6QZ82SR5TzcIv9AN4pybS0BQ5unB+OPpwVwevQou'
                                                'Ra8io4UHnrXwYO/S9SBo7wAvFJy9fTNkbd9Mtbbe+JOtN7t0yI'
                                                'aBHbLpwAMfFFzbsgcEea8cioKzlx+EnOUHSd/ZH4d19me/gwUg'
                                                'KD0pANcmBXBr4nH4N/E4Gc4E4tIzgay/dxLg3kmKex6Egs3Np0'
                                                'HQiPYh+H77EE4+r4IgO4tQFBwXeQ6mR56jLb+HopD8UQZb1pZR'
                                                'QqUWv67U8u2My3Az4zJZbwnDrlvC2BBbActjK0hT0wfb1fRhzf'
                                                'kR0HpuBPl17YdDu/bjvI9iIPejGCrr6IKCkw7PBEH6CwNx2IWB'
                                                'nFXzPQh6NXowCs77bSUc+G0l1e90x7qd7nKf1oHX/62jB+c9UX'
                                                'BSfCYIyjAOQcHpDdthTcN2ilzvg4LTX+TAmhc55BExFD0jhnLc'
                                                '6gMQv/oAOVzywz6X/LiSD4Ogcv8AFBy1pRDGbymkYoPkPUMg+9'
                                                'meBF/bk+R6MggFR+tOw8e606S7EYwRN4I5xV4Fg71KjXdC8Nmd'
                                                'ELbIN4GgP093RcGam38rQm2p64F/1fXglt29QJPTi/yO9UL/Y7'
                                                '04qZ03zGvnTSnaPvi9tg/n2Q4DQQUpTig44WE0xD+MpiRDf5xt'
                                                '6M8es78CQVGRA3FC5EDWXE8Gi+vJlF3rioJbV6aC5odUmhfsjn'
                                                'OD3bmxTzo87ZNOhdM9UbDO60cI9/qR3n0xBAW70jYQVKTzQcF5'
                                                '9bsht343WTX5YqcmXy5P3w+CXhj88FeDH4+oyYfhNflk99ofbV'
                                                '/7c6HxGBw1HiOrUYEoOGZ0MQhK29sKgkwubkq1i5uauq4DrljX'
                                                'gTUd1ypCtUu0QsHVk0xK+SSTmvC7NQo2/fZaKf3ttaod2AMFa7'
                                                '/pAYJiOvTCTzr0Yj9rD/C29qDCbQ54bJsDZ0xVIHOqQtbP+6Jg'
                                                'U9xEMMZNJNe/nHHwX87c6BEPgtZcHYCCG2k+CNo+wxUFV9uvgC'
                                                'r7FZR81Q2/u+rGY8vWwJiyNWTTzhN7tPPk5l83giCbzkOwe+ch'
                                                'bN6cBYIeP/NGwR5XdoHblV1Uvs8Xr+zz5bL0XPm9csnO5Q4Iam'
                                                'qcpmqeTVOinH6G8U4/k6Y7G4XuxQe/gSDNynsRhtR7Z63z3oEg'
                                                'lzE9FJsxPdSiPe1RcFPPZUpjz2Xq84z3UHBhfYmSV1+i2rlZo6'
                                                '2bNVf/+Uyp/fOZWjvDBu/OsGHt0K4QNLQrdZtsh4Kru7jC7S6u'
                                                'lNlij5tbJL9vCQNBoyb3RcE5P42D3J/GUf1sZ6yb7cxZtXEg6N'
                                                'XYASjYb/+34L3/W3rcNAh/bhrEFvuXQfv9y6jxMzd89pkbR89c'
                                                'DZNmrqZ3Oz1QcNqhDFhzKIMqJx4EQabQQW2g5WUJvHtZQk0/at'
                                                'pAdunFNtQSpaqaiapidewWCCr8doFq+naBknxDeu4NM2kyXxhN'
                                                'P77QpQ98CYI0zQXhmtcFZ5wOtoAgzcYWnSmjxXipqB0Kduk1S/'
                                                'HrNUutLLJEwU0TjihCbZrSBX+f0oX1v5gV7S9m9faB7lhzoDs3'
                                                'PbECQZG5tvhBri0naVwgUeNCXpOl70y2Z015MLReDqZocsSPyJ'
                                                'E1I8aAoPTH/XDd437sseRTEJRU6YJzKl24dNlsEJQw+HOIH/w5'
                                                'NXdb0wb0GcvbUNLXSW3AXPcjPKj7kTxujW0DGV/ltiG/DZ5twM'
                                                'OzuE1bP2sDNo4XoJvjBUpYfEWNW3xFsQi9AYLiuq9So7qvUkqX'
                                                'PgBBLgFdVaEk/fkcBBmiVp7VTFwZkbv7TxBkCK3SCWOpSYOCNb'
                                                'WfKEItvNERBRcW7FPyCvapLT90xrc/dGaz6SdFqAOfdEPBkRst'
                                                'QFBrU0/8u6kn1z7rC7ef9aW4Y73x82O9OeaVP8S+8idzYpVSm1'
                                                'ilOnQ6TwI067u1odzjhW2gtEtwG7Kx290GClOmtqGSMeuodMw6'
                                                'KCxe0oZ0jxaQ/tECiL6c0Yayhn3UBgoW7GlDzc+HUtPzoZDy6f'
                                                'E25NK9Kzl07woZ80ywfp6JrEprVE1pjWJRWgWCzH9sUoXi4F4P'
                                                'glzQVRVK1oVGEKTf0Neo2dBXl7OxGQQZVu/TmVbvM1pv/Q8Euc'
                                                'weqQg146UFbnxpwebX2xShup7ohILNK6oUocadumw0nLqse1HQ'
                                                'QAIM+5N1+v3JxoC5T8lv7lMwVXymCLXE6g6VWt2BOAuTIlRfV8'
                                                'kirhch8vv32pBX/QnyqD8BCa6+EO/qS9NH76X40Xsh68pHbejP'
                                                'lRnUvDID9EpyG6r+ZDGVf7IYMgrWwfqCdTSv6BNKKvoEvJfuak'
                                                'OVY0KofEwINK852oaCJtuS32RbsC5X21CepkHN0DQoFgGVICjv'
                                                'xS4158UuxXD5LghqSopQzUkRSvPqpyBIbzffqLGbr7NY/BqE7L'
                                                'c5OkO3OUbvbfbksc0egnYWkADzA7MqlDHvEUW+R9Dos1E1+2xU'
                                                'Guyvk9n+Oph926lCST71kARo/Dvo9P4djG7TnpHHtGdg3jRcEW'
                                                'p767uksb4Lmk3HFatNx9V3EeXUElEOpZ9r2pDO4lQbyAlyh+wg'
                                                'd7JbvL8NJFmPg3nW48hg+pFSTT9Cxpk5bejjZ8so+tkyyFq7pg'
                                                '3NGxxHSYPjwGPZjjYUfVRHMUd1UHD8cBsy5PRpA7l2Z9tQ8sev'
                                                '1KSPXyk2+6+AIIc3+arNm3yl1N6WBeZddQJB71e68YhKN7x9aQ'
                                                '0IyukWzgIbO1XC806VtPGowhlHFTTb34UH9nfp1cZH1LTxEeQU'
                                                '3DeaCu7rloztwALNI9MUob7+vQcLjOvfGwTd/7gf13/cDw32Uy'
                                                'HFfiqdsO/Pxfb9UfM0HiyextPcBe48b4E7Vv6aDhW/ppOl2YcF'
                                                'JvyzBwQdvhzABZcDMHlrESzaWkQ3VwWywIIhJ+HokJOEfUNZ3z'
                                                'cU87adgwPbztEHtWEcWRuGQZFXISTyKlX9HsECb1+7CYJghI4F'
                                                'Or2VPv/2FtVfky56TUHj6rtAq+/S8lZggY/v3QdBc58gz3uCmB'
                                                'DwCATFlRxvA/WR1lQbaQ0xVVdIQItfvtrkl6+MPfmQBOhHtFOF'
                                                'kvDtGxJgmpiuE8Zr69pz5br2mCPFN0tnUFcldOa0hM6YlHhHSU'
                                                '68o66P7cECY6gnRFNPWrXEntOW2GMchYCgrIt/tgHTuBu6nHE3'
                                                'jEM6WbB3JwvM0GcoQm0u7cICCz2fKKWeT9SClT1ZoMd3fUCQY6'
                                                'oDC9Tf1YEgmyv9WKBNTSxY18TS3oSBLNCw/TtI3v4d7Znvxjnz'
                                                '3dCpcDU4F66mfnm27JRni+beLiDo2Io+XLiiD2pqh4NF7XBaeM'
                                                'iZkw85Y5RuOgj62GcQC0yeuwQEPRjgzgKN99bCmXtrySbLiwXa'
                                                '9N8K3fpvpTG7fVhg5OA9MGrwHhoxUrLdSD+M6pIP47vk09cvB3'
                                                'HCy0Go2b4M/t22jDpWuLNFhTtav1sHgvyHDGG/IUOwpf02EPS2'
                                                'vy+39PfFoFt7IeTWXqo8Kdn6pB8a0g5DatphyR8BLHAEngBBFa'
                                                '+DWKB1zzPQuecZ2nkkhLOPhGBsHxMIytzpywJbF+bCu4W5tH+Q'
                                                'P+cO8sfkfwpg0T8FtD8mkHNjAjFmteTN1cV0Y1QwV48KxvLgs1'
                                                'AZfJYGvQ1h17chqOl7Dv51PEdz52h53hwttoy/BIJ2GMM42xiG'
                                                'hU8qQNDmB+Gc+SAcnX6vAuffq6guS7pkVjC6fGSEgR8Z6TmEss'
                                                'DHvc/D097n6YNCLUcWSmf0uQyCst/I33sThlmHrsL2Q1fpfLcI'
                                                'Fli+8zqU7bxOVyx1XG6pw4Q+t2Bmn1s00KxjgXledyDf6w7l7l'
                                                'JYYNP9WhA0yTOcBbp+XQkuX1fSVSWCK5UITB1VDYL+G6pjjZ8O'
                                                'rY/egs5Hb5HeQmGBqUfuwPIjd2gCKRxFCrpOvAsDJ96luZ8Dz/'
                                                'scsHxfHVzaV0drm4DTmwCtj92Hbsfuk90sZIHzvn8Ign5wV1hg'
                                                'gs9PIEh3X2GBpefugqAfU4AzUwDrLevhnmU9fWyFLFBn+QDCLR'
                                                '9Q83Jkga7lD0HQulfI6a8Qmz56BILajdOzZpxe5r0BHAsb6Plu'
                                                'PQuMafgZBCXgljbQWvUBtVR9AIaCfSQg3XUIZbgOAcfMQnLIlB'
                                                'xxvRsJmPXaSEmvjaC/+UQVivfFchKQ1+2YWtjtmBLb5TYJML/7'
                                                'VhXK2IsPSEBSamfVkNpZ+bK5kRKaG0H/wNZoeGCre7X9NTVtfw'
                                                '0G83yd5tF8o4XxbxJg2uOkFO5xUot92rNAlxnJis2MZPUIWHIB'
                                                'WGLT5DylcXKeOty2Mws0n7ipCPXFka4s0G/034pQtT2k2/TogV'
                                                'ZdeoB1lx50Ae24DO2waf0gaFw/iBqcmMxODC5P6lShxN2pIAG1'
                                                'k/erpsn7legOd0hA9b+fqeZ/P1O8c80kIMeuyWiya9KlnnlOAj'
                                                'Rjss6KiOrIZhKg379T57J/pzFy0j8kwPS5j5LzuY/65cv2LNDm'
                                                '/VRFqDFvLFlgVK9CJaZXoRqtdmaBcQPuKVED7qm147qxQG1AOw'
                                                'gKaEcLD/bg5IM9sHy2HVTOtqNrlXZcWWmH2qtuIGixyZ5TTPZY'
                                                '/oEWTB9oyXmlIwssfBYJgkb068cC9SUxoC2JoYwlDW1AP/GqUe'
                                                'h0KS9IgCZt+2mNYXtY/Os/SIB+kEnnMshkjH/8LwkwdAQlqSOo'
                                                'vQ53YLvDHdBmZLriMDJdtT3wHtsdeA81l04pNpdOqatndOG0GV'
                                                '2waekjpXHpI/XfX7uxwORSC0gptaCOup5soeuJftfsQRDE9GKB'
                                                'URovEDTqfQcW6Dc5ArwnR1Dpf44sMKb6QxA0MKMfC3RZ+D8Q9N'
                                                '+f/Vnztj9mTZgJgkpDBrLAmLhkiI1Lpt0TXTlnoivqa6Un1C6n'
                                                'IWEa9g7TYE76eMWUPl698KUFl31pgUnaTCVZm6k++NiKBTZ9xU'
                                                'rLV6w+7WLNAk3XGxWhTk/rzgJjQq0gNtSKnlb1ZIFRg/pC9KC+'
                                                '1KuhF9s19EIPXx/w9vWhdWUOnF7mgFn/AxD0NLkvC/QoHA+uhe'
                                                'PpUDtnzmvnjDH+0yDWfxpN+NyFoz53wYQniRD/JJFw+0DWbx+I'
                                                'VrcWgaCyo64ssECTCnmaVLLb48YCb0dI/olYQx/M9eDIuR5Y6Z'
                                                'wBVc4Z9Lm7F8e5e2Hikc2QcGQzzSq0ZoEmmyal2qZJ7TjIhi0G'
                                                '2WDt0S5Qf7QL9Y+zZZc4W/T41hm8v3WmqTN7s8DGaD8QFDOiDw'
                                                'tsmjoMmqcOo8NNfbmgqS86rJ4IgurmOHP9HGdseRwHguIqXFhg'
                                                '7bzZIGh760AWmIaLIR0X07XOg7my82B0cFsJvd1Wkv9bN/Z764'
                                                'YuE9eC88S19NMFD6694IG5/0/R3YfVfL9xAD8jhBCikCQhoZJK'
                                                'D+d736czIw+jEWvE8jCamYXQzMMZIcSSbI0QQjOzTD+azve+Q2'
                                                'iVhBA9OCmVOraGEcLv7o/XH11c19f7+/l874eL6+K8C44576Lz'
                                                'y1xZ4JrUH8GQ+iMZ2wxjgdHF+2Bl8T46vG44CwzPTIGwzBTq9c'
                                                'CN7R64YfXUo/Bw6lGy/rcnCzREjQRBaRd7scDQqaNBUMeF9my1'
                                                '0B51M0Ng9MwQIpODFAzJs2seCBrp2Z+9PPuj6e1SKHu7lP6aNY'
                                                'BzZw3AuqProfroerrwxUDO/mIgJgZtBkHjJw5mgcnz4kBQZPch'
                                                'LLBgcAIUDk4gN9WVBVr8/BMImjVhGIdNGIYWWcnQNiuZLvBwzu'
                                                'bhmLLrEBzedYje9XFngXGOx2Cb4zH6ZY4Hp83xwPlfnYAFX50g'
                                                'z4QRLDBjxSn4Y8UpenvKk5tPeWKZ/g8QNHVSPw6Z1A8jfBfAYt'
                                                '8FdC2pPxck9cckQxQIuqEO4CJ1AFq3/h4EpfBAFugcsQUEzUoZ'
                                                'zGEpg3H08h0wZvkOsp07hO3mDsEgl90wxmU3UduhLPD5tiTZg5'
                                                'LINX4YC3Q+uB8E9dS4scDEiMPwY8Rhcgl1Z4GahmPwrv4YmX/2'
                                                'YIEa11/h3ZBf6fqVEVx0ZQRaDfkdBI00ebKXyROTG/4AQZ/Vju'
                                                'TQ2pFYHXUWBK0s92KB1rl/QrfcP+n4JW9Ou+SNw+qM4FpnpMV2'
                                                'g1hgbkwsCFr6bDBHPhuMLoadIKjg9yEs0GZQIgj6avJQXjx5KI'
                                                'au/hlmrP6ZnIqHscBiwwEQ9Au6cRq6oU45AnrlCO3b7c7Ju91x'
                                                '9JnjMObMcfr+hgcbbnhgTI3MazW/UsHrESww6v7vsPL+71RkPZ'
                                                'IFRiecgdUJZ6jIxosFFnU9B4Vdz1FzO28WUj/OS/04T4sbvFlg'
                                                'wTIVBA3J9mGXbB8MnpENk2Zk05Ato9hlyygssrkEgqYF+nJIoM'
                                                'xnqZfhdeplCvhpKGt/Goq6hXtB0ATb4TzRdjg6hx8El/CDtGqd'
                                                'G0evc8PULqlwrEsqfVrozqGF7hjxbRoI+sZyBEdajkDNoZMgyM'
                                                'XNkwWat6aDoHu6kVyiG4kxozJg86gMMgZ6scDmk+fg3clz9IGn'
                                                'N2s8vTH6yXlY/eQ8ZXbzYYHVb1QQVF/twwLtS7PBoTSbdvwyiu'
                                                'N+GYUlOy7BnR2XaMMcXxaY0vMKHOh5hQ508mOBdatyQdDRU36c'
                                                'esoPLTLyoFVGHk3/yJ8FOl4rAKdrBTTL0oPDLD0wXPkFZiu/0B'
                                                'kYwRkg8+9nv8HKz36jx+GeXBfuiWtGnYboUafp48iRPDFypMz7'
                                                'GSCo3xIvdlzihdFjMkHQhpneLNC4NgvOr82iHX4+HOfng9EbCQ'
                                                'RB+1Es0DDjAgjakj+KY/NH4Zo2OSBo4ve+LFCz5QoImjnUj8OG'
                                                '+uHiylwQtCbPj9e0/BsN23zoaZtPcz/3Z4HOrtdAUFa9Pxvr/T'
                                                'HV/jocs79OLl8HsMD4xiJIaCwiv9oA9q0NwPTjNyHj+E0acXAk'
                                                'exwcicPW/A8EvfnZi5t/9kJdRabMaJm0JcabY2O8McLKCPOtZD'
                                                '7/XM7jcx+Mbc8gyGn4KBYYVXIBBNk/GcUCi77LgcLvcigjxZcF'
                                                '2r+4Ar1fXKHsID8WmDPuL7gy7i96+8iPmx/5ofW3+WD1bT45fO'
                                                'vPAu03XwNBN1oFcFGrAGxadh0E/bExgAXGB96AhMAbNK05gEOa'
                                                'A9D47CYICvxay7qvtei8vRhcthfTiztaFhjV/g6sbH+HGv0UFj'
                                                'gs8i4IKt7kwwLtdjL03smUFDyKBWqmXYR3IRcpqLPsH519Mbcx'
                                                'BwQx+0r99cXn865C07yrlP+FHxd84YeN6X/B3+l/kcsH/iyw+l'
                                                '4+CNqR4M9xCf5oUXkNBB3sHcApvQMwOPc6TMq9Tq9+DuCmnwPw'
                                                '5K4bcGLXDXLvqmWPrlqsg1tQA7foskHLOQYtpt8oBkHKYy1rH2'
                                                'sxdcIdEGQ1QWGBo0/dhTGn7tLBYwqnHFOw+XUJCJrTrHB4s4KO'
                                                'I++D88j79PtE4PSJgM8/LYUXn5bSogpfFphUdBX2Fl2lNRvkfm'
                                                '3wQ3PPPPi7Zx61tvdnC3t/TPMrgGN+BXTqV38WWOdfCPX+hTTQ'
                                                'M4CdPeU+2RfBKfsievt7ADf/HoBlphsgKG+glgV6bb0Fgrokat'
                                                'k6UYuLbW/DItvbNPGNlgVa77gD3XbcoeczFRaY1nAXBK06q3D0'
                                                'WQUTve6BIJ+OwAJdvrwPrl/ep10zgONnAKZuK4Vj20rpUSpwdS'
                                                'qg794y8NlbRoZ6YIHPk8qhMamcMoYiCzRtqQBBG3P8OSbHH3N2'
                                                'F8KV3YU0d1wAC7SPLILekUVElwNYoLP7TXByv0mD/bXs4q/Fkz'
                                                'dvgSDH41oWGBZ2G2aE3aZBnRQWmH3tDly8dodOLlZYYKhLCcxw'
                                                'KaE+VxS2vyL749f3QFBZL2CBiQfug6AJEcATIwCzjaVw0VhKdB'
                                                'pYoNdfZSBo2wtggZaXy6Ht5XJ644Pc7CP74OkKgNMV9Oly5NDl'
                                                'iLk7H8CVnQ+o6lfk6l8Rk2abYO9sE5WakMtMiIb+lbCufyWNma'
                                                'RlgUF9imFin2J6+aeWm/7UYkzKbdiQcpv+clA410H20U534Wyn'
                                                'u+S+VmGPtZJnfgkISritsMCCo/eg8Og9mu4CLDD81n0QVLpC8q'
                                                '0AnPh3KQhaRsACT74sg1Mvy2hpK+TIVogpjeUgyFGPLNDxfgU4'
                                                '3K+gq2uRc9fKfvu/B+D0vweU9j9kgXEbTLBzg4ks6pEFaj+sBP'
                                                '8PK8m6j44Fml9Wwt8vK6nNOB1bjNNhxKGHsODQQ+oWpWOBITrZ'
                                                'f3VV9P0OhQ075L4dLIETB0uotFrhsmoFI8vvwbLye3TcBzjNBz'
                                                'CpXSkktisl/UZggaP7lYGgD/KBNfmAyQPL4cDAcvLsjCwwo08F'
                                                'CBrwMbLzx4ixrR7A5lYPyC4WWaBl6QPoUPqApjFyCEveYyZwOG'
                                                'airs+QrZ8hxs+vBEHbB+g4boAOPWwfgqftQ3oRrGOBRnoIgjZ9'
                                                'p+OY73RYNrMKBM0/omOBXv9Ugc8/VXQxV8dC+nW19Otqutyg45'
                                                'wGHWreVMO719W0O0TLiSHy/XkWQ2fPYvrolp5H39JjRlcznO1q'
                                                'poRtcr7bFNTsKYF3iSVUfk/PZff0GOFqhkWuZpoU6cECNftPwL'
                                                'tkmTcn61igV+ZDEJS3W88CQ50bYLpzA1k/1rPA4HlmmDTPTF0U'
                                                'T7ZWZB41poOg5h90LDD6eBUI+mi//Hn26zF7bANcHNtAq+r1HF'
                                                '2vx7gIM+yMMNN3ZSN5TdlIjJx8FhZPPkuJeToW6GyoBkH3juq5'
                                                '5Kgeixc2gKCsBj0bG/QY9LUZBPWc6s0CF4dnwZLwLEp4r2OBIa'
                                                'MfwfTRj+jAb3oWWB3TAKaYBjKZ9SwwcakZBBXcG8ACc2Z+D4K+'
                                                'mTKKI6eMwuiZF2HlzIvUeiewxU7AuDFlsHNMGQ0PCuRhQYFo+K'
                                                'QGNnxSQ8FT9CywyL0eCt3raXSm5M2U93+yAc6ebKAvH+g54oEe'
                                                'ndEMg9BMy//Wc9TfeoxaZYaVq8wUFjGQBdYFbwZBrWpGscCcuT'
                                                'lwZW4OTbkMLLDEuhzuWMv9nBXIAmMO1oCgGTP0LLDo43q4+XE9'
                                                'zciSn7P0qM1sAMhsoJ9Nek4y6THlQzMkf2imWnl+nTzf5VszCG'
                                                'p6OIgFNnXZDi+6bKehy3x52DJfjPnkiuS7QnWvgAWG3SiHGTfK'
                                                'yWNpIAvMflwDgj6Yq2fNXD2GLaoHQeNIz0GkR99LDeB/qYH+rN'
                                                'RzZqWc31gz3Bxrpun/6FlgwWoz5K020+7xLpw43gVP7v0BTu39'
                                                'gYJf+rLAYpdcuOOSS4+k3ldLvU+MrQBBI2MC2SsmEMM9aiHMo5'
                                                'bGfinP+1KPsZvqYdumeup5Qc8CT15rAEEFD/UssHiCGe5MMBPL'
                                                '87Pl+SFrzCDIIWUICyx6uhtuPt1Nf63w49wVMt88/gsEfTUTef'
                                                'FMxOc+D0DQuB8DOejHQIxfXguJy2tpZKSevSL1OP9gPQiiS3oW'
                                                'mHK3AQ7cbaCiKj0LTJ1sBkHOjXoWWLTWDIVrzbSi2pWjql3RuX'
                                                'cSCHryxI/NT/wwdE8+zNiTT8lSbwSayx5AfdkDWnc8kAXGnqkF'
                                                'QfkrJd9KPWoy60HQR1fk/l3Ro11lA9hUNlBetXy/1fK9TjXD+K'
                                                'lmipHnC/RabwZBK+yGcZTdMFzcLRkiuyVT13B/tg73x8Rh10BQ'
                                                'whlkgWWrTVCx2kRl5wJZYHhjLcxurCXrtVIP1ko9uV4Pi67X08'
                                                'lcPQu0qZfn1zfQ2Ud6Fpg93QyCHsjzTfL8WIMZBD3RDmezdjhG'
                                                'lB8EQYYCfxYYnl4Ic9ML6asyef9liMO6VoKgxVcDWWCQax2Mca'
                                                '2jthv1LDD9UT2celRPL/L0LDDlaQMcftpAP9XI/a/Ro9UMMwjy'
                                                '/lfO6195/9+boeB7M5080psFJh0eB8mHx9Gl6a6cM90VjUE/Ag'
                                                'X9SGmyjwp8vuIsCPrmTz+O/NMP59/PA0FWY2V+G6tgzsG7cPHg'
                                                'XSqNkv4chRiW+AAEDS7QsUuBDps3VUPTpmp6dTCQmw7K/fmpFh'
                                                'J+qqUaJz0LfL7uMQhqWK5n83I96n6rB/1vcp+PyX0+psfMiAbI'
                                                'iGgg18t6Fvi8rAEayxro6xI9Ly7Ro8VAMwiaJecdJued8YkZBF'
                                                '2Sepkj9TJC6uUiqZcz5P0LTFpnhh/XmelgWh9OSeuD4Tcnw/yb'
                                                'k+nzF64c/sIVbaYlQbdpSfR7kBenB3mh5vI5aHX5HMU/92Mh+1'
                                                '++7H/5BF8pLHDYgBJwGyDz0XZk++2IdjUPQEi/0Um/0WFGZjWc'
                                                'zaympWmBHJkWiNbnaqHbuVoa4CLfg4seDQmPYV3CY3JZpWeBoX'
                                                '/Ww4w/68kuTc8CfSMbQNCXcr8j5H4ny/3eK/d7pPQzL+lndYPN'
                                                'IOiO5Be4Ru77Ornv1lLvBcYsMcPmJWaKlfwCg+T+j5b7b3vcnu'
                                                '2O22P6w2kgqGbXUBaYNm0vCNIe92KBLlZ/giBPN38WGLu6AATF'
                                                '7lRYYPiREph7pIQKUpAFFnmbQJC2VscCfeurQdCZ9EDOSA/E3N'
                                                'u1cOV2LemG61lg3NHHIGjqaj2HrNajx+V6cLtcT51O6NnqhPz6'
                                                'qgYQdPCqnlOuSv2uaQDnmgZ6KvkFpks/z5B+Plm+t2D53rTTzK'
                                                'CbJuct+QUmfWMGQZmSX+Bzyf9C8gcc7Mvag33RdGsmVN+aSSuH'
                                                'DGOBxvHJIOjz1t4c3tobEyecB0GfL/Dn8AX+aPfhNej94TX6J1'
                                                '32sXQFox3vwWrHe/SvzHeNMt8lbjTBjxtNNPa5joOe69DQ+xEI'
                                                'upgZyAKzn9XKvldLWk89C3TOfAyCFko9iZB6klFUD38Uyff8q3'
                                                'zPv0q+dQ0gqFDqi8BEqS8JUl/S7+tZ4JrhZlg9XO675M+R/Gap'
                                                'N4IOSf4UyZ8WaYZjkWaqlvwCk6X+CBq+y4GH7XLAoGNzQJCLOo'
                                                'wFGrwOwDqvA+Qyy5sFRm3MgmUbs8hxvz8LLG5bCIJW3ZT95qaC'
                                                'hr2yD+y9R/F5yAKjr5sg6rqJ3mkCWWDyuEdwYJzMI9mBLLDZug'
                                                '5eW9dRz1HyfY+S88p7DLq8x7TIoGeBjaX18LS0noplfhF4cpP0'
                                                's00N9OIveR9/yTzzdwMIWlCqZ4HDPGS+8JD6JvXOS+pdcqjkCz'
                                                'VTneQXmC7zjiBrqX8CdVL/QOqfzdp+LDB4xkIQ1P3j4Wzz8XD0'
                                                '6J4Cgk5keLPA52eMIGjtTX9ec9MfSy5L/suF5PpMYYGJXWX/6X'
                                                'qfTlUgC1zTqxJW96qkBZ0CWcj9fQTxqx5RUK70z9xAjB5eJ+dV'
                                                'R7cDJF9AS97HIGim1PMwqed1NfVQX1NPR3/Xc+rveoyOa4CVcQ'
                                                '1kmy/1IF+Pjs8awOFZA/Uqk5/LpL94muGip5lSJb9ArdR7QcOf'
                                                '6HnYE8m/TPIvM5O/5BdokvwVkr/zPEcWGNS4GCY2Lqbge8NZoK'
                                                'b2EAhqaO/D5vY+6FWugqCYtgEs0MtwHXwN18mqO7DA7E33QdC4'
                                                'p8hBTxFPzqmE9DmVtK6X9OtegRiR+ggWpD6iC9cDOft6IE4cVw'
                                                'fjx9VRlE7mQZ18708eg+eTxzRqs559N8v3+m897P23nr46LfX9'
                                                'tPTz3Q0gaESBnj0KpB6+lPnqZQPlSn6BE71lHveW862V862V+W'
                                                'qmzBczzfSN5I+U/CeXm0HQPMk/X/Kv2SDfywYzffhhfx79YX9M'
                                                'i1gG6RHL6OUcN26a44bO6UdgUPoRajPLhy1m+eD89wQL3hP5+g'
                                                'WwQBufIujpU0ROnsAC7V7eB0FfttFxRBsdOqZVglNaJY0fGMgC'
                                                'HYsegUPRI+p4N5Ct7gZi5Pw6WDK/jppG61mgxbvHIKj1Vj1bbJ'
                                                'V+3VwPggaekf5wRurfzw2w8+cG+vCazDfX5HzfNICg5eXy/sql'
                                                'n48yg6DVkl9g2CwzhM4y02nJny75U6LMcDjKTDskf5zkt95ohm'
                                                '4bzVTc14kFllxaBYIca91YYMmSo3B/yVHSnPJhgZm9s0HQ8UUB'
                                                'nLYoABNriyChtoi+CQaODJZ9+ItSOPFFKXWxk/3VTocl5kq4b6'
                                                '6kfI9ALvAIxJzXj+DK60c0pCKQXSoCsXpdHTxcV0ezxsl9H6fH'
                                                'xZ3rYUnnesraLvvNdjnftnK+bRvobobsPxl6LDso+8vBBppcKP'
                                                'W9UOr/+wYQZFsh979Cfr+/Gfz9Zb6S/CbJXzLbDIIaJb/A+BVm'
                                                'EJQh+QWmSf5jkr/qqRNXP3VC31ZrQdDDCHcWGDzgOAgyN8v9b/'
                                                'bB0e4XQNCdvQEsMDvxBlxMvEGvvwYWGHujFAQprtLvXHVocn8I'
                                                'giy0gSxwvnMNCLJ9FMh2jwIxI6kOBB3/WM9pH+sx0kHyO9TTvJ'
                                                '1yP3dKPmvJZ91Aa87qWaDlsQYQFHJd+uN1PTa1MoOg85JfYJJW'
                                                '+pvWTKPr5H7UyXwfboYp4WZyk33HrWXfWmmG6JVmKpX8ZZLfN8'
                                                'YMguIvDmCBdt2+B0FTat1ZoPFaGpy/lka3x43i4nGj0Fe5CP7K'
                                                'Rcr/K4AL/gpAo3ITzis36a9twLnbAHP8y+CKfxk1BehY4OilD0'
                                                'G/9CH9NyaQn4+Rej+xBgTdbwhkgdWn5fxP11HJJ3K+n8h9dKsH'
                                                'QW/i9dwcL/tKrwa400vOV/ZJO9knrWSf7CD75NQiyV8k/bGt9P'
                                                'u2ZgqV/VJgHcj8A2Y6JvkF5syReXuOmb6S/Islf5jsm4I0T2Vf'
                                                'eyr1UvJflPxv45y5Oc4Z7fdsBEGb53qwwJAlJ2DKkhOUmziKBR'
                                                'pHXwIafYn2vAzgxJcBmPLwJhx4eJMqjgELjDxUBoL0H+tYoM3p'
                                                'h2B9+iGlfBLIApOiauDHqBpa/jSQo54GYlReHUTm1dGTaTLvTp'
                                                'P6pdSDoA279SwwaEADCEr7U+7Hn9LvTku/O91AE27oeeINOd8O'
                                                'sr91kPst+QW6BJrBOdBMbR/LPvBY5ol50g/nmSlN8gvURks/iD'
                                                'bTIMkvcOImMwgaP2UgCwzrsBkEJd73YIFBbU7C+DYn6U75KBbY'
                                                '9FEOCPJy1rLAmI23YMPGWxR4EVh3ETC5TTkcaFNO/Wbr2HG2Di'
                                                'f+8xAEtQ8LZMuwQPTaVwOe+2roYVMgC1z8sA4WPayj+s/0LHDY'
                                                'xHpwnSjnv0fOf4/Mw8MaYMOwBnKXfd5D9vnkczLvnmsg5abMSz'
                                                'elf3UyQ1knM22R/T5W9vsQ2e9DZb+fL/kFxsw3w4b5ZqqU/CbJ'
                                                'by37fjfZ94Mkv8AiyX9T8v/YbRALtF4WC52XxdLOySNYYFHCby'
                                                'Co7UBfFmgZeBnaBl6ms5O1LHCNYzEI+q8M+HkZYOjCcpi+sJwS'
                                                'luhYoMWwKhDkviCQPRYEoim7BgS5vg9kgQWv6iDvVR3lz5L9dZ'
                                                'bUoxn1cGxGPc1N0rPANd4NsM67gX4zyj5plP6oNoCb2kBDbsl+'
                                                'cEv22a5mENSnUs/2lfL7P5J5/yMzkeQXWPKF1L8vzNRL9n072f'
                                                'erJb+gRZJfYPBmuT+b5XsoHMQCIwu2gaC4CyNYoL3t7yBo0SJf'
                                                'FtLvroAgx9VaFnjyz2IQ9PYFcPMLqf9XyuHElXL6eZ2Ok9bpUL'
                                                'uwCvwXVpFNZCALTH1UA4cf1dDYNvL+20g9tn4MidaPafccPSfO'
                                                '0WNuRD3kRUj93yv1f68eR0MDCHIkPQtMudgABy42kHWx9Pdi6e'
                                                'c2ZjhlY6bjkj9N8ieONUPCWNln6iVvvdz/hWZwWmimYMkvMGO1'
                                                '1IvV0v8kf5zkL5b8NyV/WsxgFpjeawf80WsHHfLw5BQPTwxPSI'
                                                'e5CelkecqXBeoGXgVBq45oOfqIFh0n3wZBn1sjh1vLvDOgAgQF'
                                                '7JT6v1OHmkNV8C6lirK+DWTjt4G4pn0trG4v+05H2Xc6yvMHPY'
                                                'abgx7TmC/0LDBpZT0Iar1f+v9+uZ9jGqBwTANtZ/nzsszzVxtA'
                                                '0L+Sv1HyN9ua4bWtzLsPZd59KO9nnPT3cWb6WvIvlvwREWZYEG'
                                                'GmzZJfoOE7uR/fmek3yX+y5fy3yLy0Rer/HzNByJwbR2kZcVD2'
                                                '6wAU7JxnAKc8A5nbDsYnbQdzycbtIMjeRUHBse53QZAze5Aje4'
                                                'CHzSESoCuZ1IIuVidQdnUCBDc2waTGJjLEvdMK47LLTig4suo7'
                                                'EFS3IAIEndiwmdI2bIaoAA0uD9CwTp2keKiT1JLtzig4etdGEJ'
                                                'Q4xBt3D/Fmr6Lz4FN0niYWtUbB1iU7FKH6+g1CwZZ/x0KHv2Op'
                                                'qN4bBZ+MVOFEpEou77Q45J2WI3+6A4IqCgej4Mz2O+F8+50Ucs'
                                                'gHpx3y4dC/GQTVBSsoOPbSXYi7dJcqwnQo+GT9QxCU9+53yn33'
                                                'O1T3sSZTH2tICR1FApw+kN3pgwNgmNVNK4x/v62jxrd1EDErqA'
                                                'UZ3+xpASn/nm1Bmds1LaDs/YsWlN3xjtbU8Y4xZm03FJw9rhfk'
                                                'jOtFOZOd8PJkJ9a8i4bmt9GU1te1BdwfmEolA1PBcORPbfiRP4'
                                                '2v51RR85wqsBw/rQVZXPuhBTQONbYg61mPVaEU3X8Nhfdfk+Hz'
                                                '7krK593Votvd8frt7uzi1B+GOfWnaL8BKNglex0Iyts7HAVP7H'
                                                'kIgnoeIsfXNopQg/9XTgKSus5tQcX7tlLRvq1QNPFCC8q8eUNN'
                                                'v3lDsT7wDgSF9wlQgvsEqNXOPfCRcw/OvOEC6TdcKNXeGY/aO7'
                                                'PZfQM0um+g0JFuKNjS5ggIinnphZteenGU03kQFP7HEhA0/suN'
                                                'NPHLjeAScqUFxX93XhXKC+cPULDJZ67S6DNX1XzVEwVnhnqAMd'
                                                'SD1rxyxrWvnLno7xgo/juGEq+64Z6rbmy2OAr1FkfJPMMbn8yQ'
                                                '+/NdFqR/l0WZOX54LseP41vlg6C6wHwQlD3jiCqU8H9a4ef/tO'
                                                'Lg0BglNDRGDU6zRcGx50ZBzLlR5Fw0EAVb798CglI/dccjn7qz'
                                                '9vExEBSb5Y1bs7y5Ot8IglLb+OORNv5c/TgfHj7OJ2OdFtU6rc'
                                                'xkd+DmuDu06JQFCjZ8+JMS/eFPaodyOxRs+Bsg+m+ggpRBeC1l'
                                                'ED/33wZP/bfR4Sp3FDzxQhqMv5BGi3v5oOCJ3QnGdyfKHu2PF0'
                                                'b7s1V+AQgK6q7guO4KW1TfAU31HQprApzVBJyRXw5n88tpbvve'
                                                'KLjx7UcgKPmrwbjvq8GcnRsHgk5+5YG/feXBITtkntpxgk5G+a'
                                                'Bg+ykMvacw2W/wR8HaY9dAUJpWQcFWq++CoFAbxE9tkL08K8DH'
                                                's4KSpupQ8OLchyAo08MF//Rw4dwxP0DBmB/o4T8eKDjl45MgKO'
                                                'yaD8685sNxsdkgKFv1R8Hx3xWCINN8BQVbWZVAZ6sS2uaOKNi8'
                                                'pwIa91RQ2wU6FBxtVQWrraoo0TcQBafZ1sAx2xoqjCmmophiiP'
                                                '9yiWr4cony/NPTJCCyTycSYFq8lQSULJ1LouX/qyMB9quSSYBm'
                                                'fKYqlIiCKyRA45cRYPDLyPKfW0/aufUQHlGoCLVbh4tk0+EipN'
                                                'h/1ILqNv/UAsL9d7ag5rhQ0uwIBZuD/2tBiZtbU9Lm1uChlICb'
                                                'UkIpX49XhRLe5gUIyv7tgtb02wVjQXI7vJbcjrM/P6cI9fXAbi'
                                                'hY97EtaD+2pfTK3ni6sjcH9Z4Ao3tPoGAbJ5xsI/UndhU0b1lF'
                                                'dqmnWkB03+4kwGbBZhLgfCiCXA5JbbfyIgE3tx2kom0HIfqbE6'
                                                'rhmxOK1YW/SIAhebxWlzzeGJdUS/FJtdDoWqYI9VQnpvRODI37'
                                                'JrSgGZGJFBqZCEmeu1rIu5lCurlTIFGf2YJind+o8c5vlMgt90'
                                                'FQiqOXKpSSppcgSDO9XmuaVm9MWmKJPy2x5JQl2Ur6kmx1ZXE3'
                                                'FJz5YR/I+LAPFS7tg4I1myaBxaZJdHObEwouWroaBF2c7YKCPc'
                                                'p+AEGFn22kos82gvHGEsq8sQTmfzOcwr8ZDp77D5PH/sNQEpes'
                                                'muKSFcMfBSQgO3a/VhgT8qopMa8arKseK3ZVj9WEzkaK72yE4G'
                                                '5TYGK3KdQUuqsF2AxMbEFGv4mU7TcRjA/Ot6Dgsf+oQgnNLwNB'
                                                '8c/7qobnfZXgmlcgSBPaXjF92l7NUNrjGaU9F0X8pZRE/KUmG7'
                                                'rjPkN3rnPvB4L+ft8HBTvWfAKCTrx3QsGmKWtB0I/kgoLjuu+C'
                                                'hO676PXy4Si4ODEFBFnlDSLrvEFgd+ooCcg+vFMVyuGU65SSch'
                                                '2ys25rhTGsqZLCmyohOP6lItQ7Vn9SidWfEL0+tAXV63ZSnW4n'
                                                '5HT9qQXldB9Dud3HQNxKtQUVrahSTSuqlDrLByDI9LidKpSSa2'
                                                '/gzrU3ZJjVV4mc1Vd17tABB3bowI6Tbiouk26qka42+I2rDZu6'
                                                'DoCyrgPILs4ee8XZc5oyDQSVLRuA5csGcGoPAxzrYSCrvkOwU9'
                                                '8h7Dg2AQSFXhiOgp//eAhe/HiIZu8cgYLTIk7BsYhTZDptUIUy'
                                                'fftNCt1+E3RP2ypCtXB6QBqnB9CU1KoFHbY8SymWZyGoajaMrp'
                                                'pN412200SX7ZDU9HMLim/EFtDcNrsFxaffVVPS7yqOukpw0lWS'
                                                'qVW1UdO6Whv1+1sQFB4wQtEFjFCf3uqAgtMdSpUUh1L19Q0bFJ'
                                                'z2eBAICunZF6f27Mtp20NBUFT1AFxeLTv/2u8hYe33NGz1EBy+'
                                                'eghXr9wNNSt3U3InN9zfyY2L9hwGQUUFI/B6wQiOVX6Hzcrv5F'
                                                'vhhX4VXpyZ9ycIctQMUIR6bGwZpY0tg8x57cE4rz1ZtzpDVq3O'
                                                'QLTyRQsK6hLbAgz3klvQ/CI/iijyA6tdF1tQ48t8VSjWS6tAkG'
                                                'llplGzKlPrtf09CAr/d7QS/O9o1XlfRxywryOH369SQu9XqaNX'
                                                '9cAPV/XgoHOuMPqcK6Ue6ItHDvRl36KZoCuaSctCnFGwb+UG0F'
                                                'ZuoObbQ/DN7SFsPpwIfx9OpMJQNxQctesICArv4Imfd/Dk1C7p'
                                                'ICjV2huPWHtzyrfnQVDNtlEouLn/JWjqf4kcXazB2cWaxrz6nU'
                                                'a/+h3qdnwF1Tu+onf/xZDmRQxEZhyEZRkHKfbMyBaQaXO5BQVP'
                                                'zFaFkrHvEQhKGbHDaBixQ5uQrkHBjutDFaEaP7dCwbrvzUrQ92'
                                                'Z1Uu+eKDh3tRvkrHYj64EO2GWgA0d0/BwElV10xvKLznxSFwPp'
                                                'uhgyeLjieg9XDsrfA+Pz95D5sBs2HHZj09ZUqN6aStZjPbHLWE'
                                                '+2fpAOVg/SKVTvjZ/qvVk7KAsEpZwZhQfPjOKJpy+BII8MPxyR'
                                                '4cfZV/PAeDWPgouXgqAfy76npLLvIXzrYZi99TB57R1OHnuHQ/'
                                                'Ouqy3I8VSGan0qQ7HJrgVBmgWjjUKbmvQBCk6p/koRak7/Tni5'
                                                'fycO7/6fIlTnrJ44MKsn+3p5gpeXJ6X84oAHf3FgX2Uu6JS5lO'
                                                'k+EAU/37dJ+sQmytzqiue2uvLzf36EF//8SM0NbvimwY1d1h8F'
                                                '1/VHybTBEys3eHL2r6fh4q+nKWS5N05b7s3xRVmQUJRFuaWjUL'
                                                'ARcuA85JDjbT/sd9uPm7T58FqbT+nLA/D35QFsaXkDBGlDj4Kg'
                                                'jJjBlB4zGMos81pQet9fVKFoKx6DIMPT8izN0/KAkrWt8O7aVh'
                                                'zef70S2X+9OrqyE35Y2Ynj498qQi34zBYFJ1V5g6CQYf1w2rB+'
                                                'XPTFFyDIat9AFGz1bDMIOlnpioI1XZPgnXUSabzc8QMvdw5eeg'
                                                'ymLD1G91VPFFy84g+4ueIPOnzEGwWnRRvhRLSRwlr74qzWvjw6'
                                                'LwcEJf7nh3v+8+PU9Hw4nJ5PITsDUHDBwhuQt/AG1ZVrUXCS0x'
                                                '3Y63SHQr4tAEHh+/arQnF+0QCCNK+Lzmuai/w9wluj4OzeW5Xc'
                                                '3ltV7YHOKDi2uRUImvjCFie8sGXNVj8QpP2tHwr22BQBXpsiqM'
                                                'JiEAq2HhMLncfE0nzfoSg4yu1nEGTxnTsKjgg/DoKaX3ri25ee'
                                                'rFPOAChnSHfLGwNveXOcg/QRB5Xmu/riPFdffj71MghK6uGPP/'
                                                'XwZ6OT9GanAkpKC0DBuis3QNCaZ1oUnB15B3Ii79CM1oCC4ybe'
                                                'h50T71N1u79BkCY5IkBkZaAF/g8tODw+QQmNT1DPh3ZBwRGD2s'
                                                'H8Qe0oNd4Oj8bbcaSzAkLO1xEFR+77Chbv+4o8vh6EI74exM2J'
                                                'W+FdosyOO4bigx1DOSxoL8wI2ksWF9yx9QV3XjwxDSImptHEES'
                                                'Px4xEj2a5VBti0yiDPVj4oWHtRBbioUlmwLwouKr0MhaWXydnb'
                                                'Hwd6+3PIbjm33QW07mIACo4acBMEXbRUUHCJegcE3bEBFDw67j'
                                                '4Exd0n3b+AgtNOl8Ox0+W02L4NCg7/d58i1Gxra2Rray75qCMI'
                                                'qnPphYItz+pAUNovjvjLL46clLYEktOWkFXxIOxYPIjrHmyDmg'
                                                'fbyPnhUBzwcCgbZ+8DQa3aeqDgCO9fYJH3L1Tw5Ui89uVIDsvJ'
                                                'AEGrPXxQcO48grx5RAUrfVFw2ZwrIEgX4o+BIf4c0eoaLGh1jS'
                                                'zLArB9WQDPX38TBBn7Kqj2Vbiu/V0Q1HMQoGCX/PswKP8+LXkP'
                                                'KDj5UTnsfVRO8Aei4OYVJhA0/Yo1CtbM7AJNM7pQN+6FgrOV0S'
                                                'BIO7A/Cs4+uRRyTi6lSL/BuNRvMOcOioMrg+LI13sYCk5bmgyC'
                                                'LMd7oODFfU7Akj4nyHBoJK4/NJLtY/4HvWP+R/azfbDPbB/2as'
                                                '3g0ZrJI9kXhcxvV2R+u0J/L/NHwc3fXIN331wjn/8CULDd/Zsg'
                                                'aLGngl97KmwOuQuCeo6SvKMkb/tSGNS+lBZ3QvyqE3KJXQUI2Z'
                                                'cQL6jIISkmmJJioppJOhTseO4hOJx7SMOm9kbBKVljITVrLNkd'
                                                '7I+2B/uzJi0KBC3aPxgFG7/cAYLstwxDwfbf7wdBNjs8sMcOD1'
                                                '7z7gQIWlAyEgWHBp6F6YFnyWa7Dwo2pDCsS2GyvuCLXS74cnLE'
                                                'VTgQcZXcdvmjYPuya9C77BpZddZix85aThx5CxJG3qKiMQreGC'
                                                'N5D0jeA3fp/lhAwSFjSmH6mFLy6oMoeP6EChDU+y9EwTZ5JuiW'
                                                'ZyJdqA4xVMfJ9x7C3nsPybFQh4JH76iGMTuqSWvrhIqtE0fsXQ'
                                                'WL966ipveDUXB22k6gtJ1UcncY3r07jJ13HADXHQfI47oHul+X'
                                                '/bTiVzhc8Stt7uyFgu2az4KgxedkHz/nw5kB2SBo0SNfFDz/8V'
                                                'WY+/gqvTvlj4JzxhSCoM2DtCi4bPstqNh+i/6eoaBgc63krb1L'
                                                '80MB54UCp28ohVMbSqnIBVFw0NoKGLO2giYWIwoe/cwEgqLm6H'
                                                'DFHB2nvHkIh988pLO3dSg49JdqmPFLNa3xDkTBHlY14GZVQ45z'
                                                'XLD/HBc2Vf0AFVU/0JLBw1FwyZ6DcH/PQQq1HoGfWY/guqyTUJ'
                                                '11kqo/9MKqD714ccY5WJJxjpyrfFCwqTgbBOna+yG29+PFEbkg'
                                                'aEGBPwq2SC8EQWmgxV9Ay88f3oKnD29RzhIFLy1RuNG9BJ66l1'
                                                'B6BKBgXyoFHyolXx9EwZGnKmDZqQqa/gBRcHWfShAUt0iHgof1'
                                                'rQK3vlXkVa7DkeU6tr4su9nlakoLCETBNgNqwHpADRm2BuL6rY'
                                                'FsOboWBAVUalhbqUHDzjBFqMOc+7LAWFMoCMo43tACNB/1ztKM'
                                                '7h3weUJPDk/oiTr9CNDqR5DG93kL0IxL1jqOSzb6HezIvgc7Yq'
                                                'RltRJuWa1aOkntdOqFmXE6EPQ4YgDXRQzAIPN6GG1eT2U/1bUA'
                                                'wzIno2a5k7aLTTu2tmmHHmtPK0Ldd6wHJx/rgZmFQ8FYOJQ+u+'
                                                'zIoZcdMXraNxA17Rtqdi9rAY21A1RN3QBl5ohWHDaiFWafiVbS'
                                                'z0SrR6u6cmpVV4wu6AFrCnrQ5cd9OedxXww6MAsmHphFMc2FLc'
                                                'A6fZcqlKZHr0hAeEh7RajbiztxXHEntNa8VYT6/EZvFlh3fjwI'
                                                'Wqd5QgbNE9Akfe8vzpt8NCwwe+YkpWjmJHVgoSU7F1pi0a4cRa'
                                                'gTwqx5Ypg1+jZZgbbJiu6/6skCM9Nktk4bSaVsz2Vsjx6rp4Og'
                                                'Q1f6c8qV/miXsgJsUlZQ24GDWKDvb7Hg/1ssvesgs2jHWtDcDT'
                                                'cKbY7TGxKgC+mpCDU4ui0L1BSeUJqunVC/ndyZoyd3RkPABxAT'
                                                '8AE9c+vBz916YJ3VEDBbDaEc9z4ssOTZRDA9m0g7FzmyQE2/r8'
                                                'Gi39eU+sKZBeY8igFBs6tMFF5lAt2IBqNmRIN2dNp/JCDb4bxW'
                                                'GFdYWXCUlQXq5scrvvPj1SQ3KxZoml6vCNViSncWmNvPAQr6Od'
                                                'DRH3px6g+90NrxQ7By/JA+1fTjUE0/Oe/5ct7zqSJrAAusbjaA'
                                                'IMet91uAo7WXKpRFb/+hiLf/gGZyB61hUgdjp20fsNW2D9D06y'
                                                'JFqN1etWeB8T/cUJJ+uKFGz+3KQu57N4gzdaN/Kmy5scIWDW99'
                                                'QdC3W/py9Ja+WBc4EwQFRDmxNsoJDVXfQmzVt1T25lYLaPxhuW'
                                                'r6YbmypH89RfavB8PZ3CzN2dwAz8q35FH5FkzjRirZ40aqM0+1'
                                                '47BT7dDQP1Pm50yVPu/CAmMntQNB3fr2ZIGO291AUMT7Piyw7u'
                                                'EnYH74Cfn2688CHbcsBfstSyk44FoLMFnsVYWS2r2aBIRf+9Vo'
                                                'uParFtY3kW59E2RP+k+rmfyfsfrjNiyw0eqAYrI6oCrjOrF2XC'
                                                'dMv/lCEeq342w4epwN1l0dAIJ+Xtibkxb2RsdRQSDoU6Och7Ef'
                                                'WlZFgFVVBIWPvdICHM+dVz3OnVfW3ConAYYb3VWhhD55SgI0e7'
                                                '7XOu753jjwbit2vtsKraduVCynblQXOnTkCIeOGD6nQhFqwrxu'
                                                'LLDxDzto+sOO1rIdr2E79FUAvBQgmwkOLDD7v89BkI0jt4DIO6'
                                                'WqUNaE3yUBHvlTVF3+FCXD9QkJ0ISc89dMP3d+7zgNC8yun6wU'
                                                '1U9Wj1VYskDNuMtKU9Bl9fIia85ZZI3xEzqBoA1tbVlg07OR0P'
                                                'xsJFXl2XN1nj2mqdPhpDqdNoyV3WjsYzBcamsU2pPrnpGQ+7dD'
                                                '3vcOY3K/tyRAM36oItTVtq1ZoEf7zYpL+81qY2g7FpiyPENJWp'
                                                '6htvu0I1t+2hE15SZFqGzThbNtumDEoTaw+FAbGrGnG3vs6YbR'
                                                't3qBIENpDxaoazccBF2vt+Oiejt0DESwD0QKuNKHtVf6YMrmYE'
                                                'jdHEzXIh24INIBIyvCIaoinFY/cWSBjsGRIKg3DmCBuVXroKBq'
                                                'HS2IGMgCLT7ZDIKavhrMAs1/xYGg4A9rWoCmu8EotI1HG0mAZu'
                                                'pArWHKQKPn0tfksfQ1ZLfvogjVWPwBCzQ9iVSKnkSqj4a05eoh'
                                                'bVG37Ljiu+y4Wurfgcv8O6D1kmJFqHs6dObEDp0xZcF7JW3Be3'
                                                'Vgcld2Tu6KOjcbEPTbUxs++dQGLU8MBosTg2lKVzsWGO3qD2tc'
                                                '/cnzTW8WWJcxAQTNPtOXBaZMDoPUyWF0XevIRVqpb60Wg6AFe5'
                                                '1YoKZ4NTTfWk2PCp25utAZLRbGQNuFMfRv2SBuLBuE8c3bYGfz'
                                                'Nkr0rGoBpqmq0TBV1T7PfkICNOEBAZrZAVmmsy9JQPbzaq0wKn'
                                                'HSb+Pk/q2apgg15D8LFhg5dq8SMXavurBXe47o1R7Db19VQm9f'
                                                'Vee8seLwN1aYHv5MSQl/pk45YM0Cw6M6g6CQNjYsMMLXCRb7Ot'
                                                'GOkbYcN9IW6957gaABw3qz87DemH19DOReH0POT+1ZoObUpyBo'
                                                'T2w/Tozth+mHFoKg2U/7s8D0P1bCyT9WUpCnMwtMi9oAglpNGM'
                                                'QCPQZvBbfBW8m3h6kFmBzfGDX932iTr9eTAMP+XVma/bsCYhuf'
                                                'kwCN369aYXz43TsyffcOTOl+ilCHZrXmYVmtUfcsTvF9Fqf+/L'
                                                'QdJz1th5ruWYpQe9R2ZJta+R5G1CpCTd3XhQVmH7UE41FLmtu9'
                                                'OwsM/cMewv6wp2Wf9GSBQRfcQdCr0F7cFNoLLe8GgqBuw+1ZYP'
                                                'irKRD2agpRkQMLtHSfB4L+HdOfG8dI/vXLQND6hAFsSJDv4VsD'
                                                'CLI/O5AFpo/bAqfGbSGPhrIWYG20VzVGe+XOjVoquVELun2BRs'
                                                'O+QK2Nw1MSoLk8RyuM2269odhbbyDl576K4ee+ath3Mq981wo1'
                                                '5nWKo3mdOvtKWxbYOOiUUjfolJp0rQMLjL97X4m9e19tn9iZLR'
                                                'M7Y+jFViDokn03zrHvhrm2tpBta0tPFvVg86IeaFjiCmuWuNLx'
                                                '7+047Xs7TMrTQnKelrrN78MCY0dMAkGjejiwbw8HLPplNgjqlu'
                                                'TIAkv8l0Cx/xJa/Y8TC0w3rwFBHg4DWeDzeZtAUGTGvRaQ7apV'
                                                'hRJ3tZoEOF7eb9Rd3q9dgNKPUfpx/t0ATcHdrBn9XlFov1eQ7f'
                                                'mBItRi7w9YoOngPEWoRxPbcGpiGwzvekiJ6HpILT/VnstOtcfg'
                                                'kEIlKKRQ/WZLJ47c0gkdE18pLomv1IX9u3JE/65oiOgKgh6vt+'
                                                'G69TYY2m4gCOp1zJbtjtli8IlRMPHEKOr5U28WGDF9HMyfPo6e'
                                                'TuvLAku6zoCyrjPo5KN+LNBU9SVUV31JLuOcWGB4cTQIctvozA'
                                                'LrVmyEmhUbqT7yDtVF3gFd6UxVKH+kV1J6eiWEjy816saXaheH'
                                                'mEmAJtLzvPD3+vwFCTC8ztUKY+fb78n69nswtApShDp7lgULjL'
                                                'y5W4m+uVu9uc2SBYYfv6CEHr+gXlphxTkrrLDx2ydK07dPVG1/'
                                                'axYYn9oRBE2P784CS7b3g7Lt/aT+95T63xM1Kz3BYqUndbzSi6'
                                                '2u9MLshaMhZ+FoKkuQeTNB6sHCaWCxcBqdce/HGe5SD1Z/ASdX'
                                                'f0FrZRdeIzux4UQUxJyIog6PBrDAxeu/B0E/Ot6iJMdbEJ//nZ'
                                                'qS/53isaOCBIT/ZqUafrNSOs963PJ39aCb1dqomdVa+y7hGWl2'
                                                'PwPds51ax2c7jdmeb0lAuM0wRajfOrbmaMfWGLxosyLUeXPa8f'
                                                'w57TC9KEPJLMpQn83uyM9nSz3wqVQavStVV/suLLAkvw0I6p/c'
                                                'jR2Tu2HRm15Q8KYXNVf2YIFJw4dD8vDh9KDRjk2Ndmj4AmHNF0'
                                                'iP8vtwdX4ftPslGGx+CaYeK2W+WCn14H04CFrw1JEFmpZEQvWS'
                                                'SEr/cAALLNKsB0GRl663gMbmHapQXoeVUlNYKZheDVWF4vNZDf'
                                                'l+VgOGc2uMmsw12uTzjSRAk+qiFcb0mNckwHCnixJ+p4taXf4B'
                                                'Cwz/Y6kS+cdS1eDZlgUaio4r0UXH1Sf6DmzWd0BdbrHim1usOn'
                                                'frzAJTjO8Vofoe6coCfT+1Aa9PbWjCKxue+MoGG4sGw/OiwbTe'
                                                '1o4NtnYYMdkfBL3U9OEmTR803Z0AgjizL2dn9sXsyDAQdDPQkQ'
                                                'Xq+i8GQd0POrHNQSeM/nc1CIr8rKAF6DwPqB6eBxRLxxISkB07'
                                                'QRVKcVAVCTAkZRmFtun2ExKguRIUILJO5b2k9LyXoEuo0Qrjrj'
                                                '0ajt8j/RGmK0K1fWfBdu8ssOjYXqXk2F71YP/2nNK/PaZ75CpC'
                                                '7dm6EwssOvdMMZ17pm4+as0C6/Z3hur9namrlQ1bW8k8+7kTCL'
                                                'rvZ8sCIwZ6g6DWnr3ZwrM3Bj0bA6OfjaFDL+055aU9Jl3/FASd'
                                                '2SHfww7ZPy4shKgLC2nAy/7s/LI/phWtBEFJD3NbQHzMKVUoE+'
                                                '8Uk4Dw0K9UoRQPNZEAzdxXRqF9XV1PTdXSH0u2ZWnubQtIt/iP'
                                                'BGhKT2odS08ay3a+IwGa1f6KUO1yWrNA09QdilBfvG7HAiO/yV'
                                                'Kiv8lSzX93ZIHhW2sVob493IWbD3dB03VLqL5uSbre3VlgxH17'
                                                'WHzfnso/7clln/bE0Fp3CKuV/jhb+uPsXuj4JhAEnfC0Z4GGPl'
                                                'Mhps9UennbgZtuO2Do5HkQMnkeXZ/Qn4sm9EePA8tAUPXsyy2g'
                                                '6SmpmmekGNffIAHxZzapQhnUqpxcWpVD/P0+asr9PsrO2lqKr5'
                                                'X+OAKMhhGgTfB8SvGeT8HQ9QutMJbUvCEBmi8cFKF+E9OKI2Na'
                                                'oeH39YpQqwtlXixsiynyroX64+0OLNB6RKki1E37OnPMvs5o+a'
                                                'QVWD1pRTkDpD8O6IYROluYr7OlVUt7cPTSHmi3xxXs97jS5S12'
                                                'nLPFDnX/aEHQ1S/7cO6XfdA3ZBJoQyZRXG8HFpiSNxtS82ZTh/'
                                                '2OLLDksyUgKKzwQgsIX3NdFUqofSEJMNzbowrlYv49ys6/B+Fj'
                                                'A1TD2AAlpbyaBGR3TDYKrS7kHxKgWVURYFhZkaUZ+YoEhGtaKQ'
                                                'ZNK7UrfsDW+AGmfDFfEerU/W04ZH8b1EUeUoT6wbn2rDnXHjW/'
                                                'FypC7fJDJ7b+oZPsN6+U3PpX6naXrhzn0hWzd3UFQT9utmGBSW'
                                                '4DId5tIE0/acsCY6+PAkF1yb1ZYMq34yD123E0b2Zfnj+zL/qO'
                                                'nAGCuL4fZ9f3w4jWi0BQqxFEmhEEKf1MqlDsTuaRAMu+x1RN32'
                                                'PK3tg7lBR7BzQvZqiOL2YojlcrSYDjjntG3Y572pQvzSRA4/jq'
                                                'T43HKz+P5S9IgGZvnlYYy2rfk4DwvCAlMi9I9VlgwQINnolKtG'
                                                'eiar/bkgV6tL6o+La+qG5da8Wxa2VevvFEEeoGF2sWGFHYEeYX'
                                                'diTnn7qzwKQz/UDQjwU9WWDuPk/I2edJr/N7scD47aNB0IIkex'
                                                'Zoip0Ggkxe/Vhg9t4vwLj3CzJtPt8CUkobVaFo3K+SAJe1Z1XH'
                                                'tWeVVN9bJCC+erWaUr1aaTpSQQIaCzqqmmsdlaRlj0mA5u/3WZ'
                                                'p/3gcEH39GAnQ/xWvDf4o31nz0luo+egvZFcOU9Arpj0OkPw5p'
                                                'jYb8zUps/mb19pftuPhL6Y/D/6dkD/+fWrGgIwts3Fmp1O2sVP'
                                                'cM6MKJA7pg0Is2IMj6SDcW6OLcGwRF18n3UNcDG6cOB0FPXtix'
                                                '+YUdlmxFELT9Rh+Ou9EHo/ODYU1+MM38zkHmVQdM6jcHBFkUnW'
                                                '0BcXc1FH9XA9ZHL5GAkqYralHTFaWk9DoJSIndoQrFIaqUHKNK'
                                                'wWA7VBWK/Tc1JMAw7luj0AZdbyQBhudDtZr/hhqdkl6TY9JrMB'
                                                '22VhoPW6tWtR+wQM2qZYpQewW0ZbuAtpjulqYI9d34DixQ53Rb'
                                                'CXK6rf7RqzMLDG6vAUHjT3RlgY4GGxD06L0NV7+3waKXg6Hg5W'
                                                'D6ta8dn+xrh8FR/iDofNs+LLDk9QQofj2BHKkvCyzaFQYFu8Jo'
                                                '+VwHjprrgCWbwqFsUziF9tCxwLKHlSCo6ZEVCg5e8lTRLXmqJn'
                                                'tVk4CUgF+M2QG/aJd2H8iR3QdiWuAmEPT0kD8LtLErBEFhP+tY'
                                                'oGNeFTjkVdHLcXpuGqfHNOt6OGFdT5bPdC0gY93PJKC9b0+29J'
                                                'X77OoOBa7uFLLdhwWGHWKYfYjpiyjk+VGIaQkPIDXhAZH1ByhY'
                                                'tzlcCd4crmqWeQSILIppoOyYBkj/7P8t3X1cznf///GGWZzNkh'
                                                'DLhFhIQhdHdRyv19Gx0JxmbWNrFmsWi7lo5kusWSfNGs1pO0Oj'
                                                'bbHmMs5mIXS8Xl1fqySEEJIuDjSLs5ltv6fb7ffH/Q+3eHd79H'
                                                'l/3hc3lJ0CpwW/ZgLrnGhvjYj2Zpd1B2jwugPicdiPXzzsp8bn'
                                                'cwmk57EBDFrTfyKV9p8oYY5TCCS63/YnKCc3X4C8n6+wuj1fYS'
                                                'q9/6IC91i7mXqu3SwbT/oosP2rJwjk3odB2vFhEHvaagjE7jdS'
                                                'YMfsK9Qn+4pMqDWrd62Zp6c2EUj6mWAFDgtroelhLTJptkV9Zl'
                                                'vY/Eobgbzxo0Vn/mjhqLntBNKF9QQ4NzDZlB2YbL1za5jabg3j'
                                                'xKiVlBS1Us79e4LW/XsCpy85TLuXHJZN+QGalB/AdY/KqfZRuY'
                                                'x0Ipy/iGPWXaIV6y7J97PNChxx4wbNvXFDqr4OVuCOebfp7rzb'
                                                '0sdgUeCG06106XSr2KUtfYLmLl8vkcvXU1vtHWmpxfmobkqQ3b'
                                                'kpp07GD1Zgl+pQAik5NE5LD43j9Fm7aPesXfLghL92nvDn5JIC'
                                                'Akn526jAaannaXfqeWkoZQX2Lmskr7JG6R8SrM4h+PpYmgkk7Q'
                                                'UPAons+9MTZEvuKx3Jfclx1n8FKPbRaQEKXbPNGrZmm+mtz/pq'
                                                '+Gd9OfGLfpT0RT951OahwD3Svyb79K9ljtFXI4y+HNf3FK3pe0'
                                                'pGZgape2YQR18/QyDPD2R1HcgcYbxKs41XJfS7TgKx+yTdCDn1'
                                                '+i8Cmd5nuYT1WU5po01WME3ZcVFCd1yk3v33i33//eQ5Z6h4zB'
                                                'lKI77tqe7f9mRvvwwTWPP6j9Dc/iM45uqnBPKH10R97DWRc5Iy'
                                                'SZIy5c/+gfq4fyDWv0qsf5XyeNkI/mPZCE1wiCeQWPMzvMr8jN'
                                                'ot+MVkv+AXa4fpIIGE/jD4CbIb8oYJrL3DL4hD+AVyzVgrbhlr'
                                                'qa51pdS3rqQ+fX4Txz6/Ua4hwQg5B397XjN+e54jd75GIPdivL'
                                                'Qjxoun3/iRQFrm+TLoiu2nCCTDZzQf8hmtbsnf0PDkb2RwbycG'
                                                'jT04gEASGqoJJObzr6xg2nJlACVdGSDe9zOeoPim4QJ0f9Ve6V'
                                                'i1l3Jc6wRoyxvLrWlvLDe9aOmnHpZ+bCgYQiCejiYe62jSrPrz'
                                                'dKz+vCy66M+gUX6FBNLUOo5vto7TqBd207wXdkvH/cH86/3BWr'
                                                '/unwQSu/Qegdhdag6Kv9R8KiXxYwIx9vjXExT/wzormGa710q4'
                                                'ey3lumUKUGTNcwK0cOpojZ46mo1nviE68400v+GtwDXGA1RnPC'
                                                'BHF/tq1mJfTjpxijaeOCV2QwIUOPZwKa05XCpflQVpUlkQ1xtr'
                                                '6ZKxVgpPmhTYHHKRLCEXZbM7K3DGkqu0b8lV8fE0K7Cb/w16wf'
                                                '+GbLGZFdj2ZxPd/bNJHFYHK3BHT7yvPW/LyWvBCpyd0EInE1qk'
                                                'eoxFa8ZYeEVqK4FY37NoznsWjl7YRlEL2+TXLy3a8aWFE/5uI5'
                                                'A5ey0asdfCcUvaae2SdmnJsSiwg7RTH2mXraXPa3Lp89yVHEad'
                                                'yWGyOGmUAkf9/iWBTJntpcCR236kedt+FKe/JylwbvExAjE6Gx'
                                                'S48ukiAvmkV5DG9Qriho3VVL+xWkaPN6nHeBO7vHyBBr98AZ+R'
                                                '1CbE0b83YPwGGXSD1eUGs/Oo6zRg1HVZfcysscfMXOjbRCBfvh'
                                                'asia8Fs3FFMwWuaJYL+cFanx/M5kEtRINa5L/9LZrZ38JZr7QS'
                                                'yNTXLRr6uoUzx7cRyK61Fk1bi/X3HNbfc23S/wcL7o8WTprRTo'
                                                'kz2uXZD59Rhw+fYe+goyawntw1UIHNW/wJ5O6GEQrsMSuexsyK'
                                                'l3ljPBW48JVUKn0lVWjRRAWOeetnWvrWzxJ8z0/N9/w4/O9cAn'
                                                'nON1AdfQPZ/j+VBNKQbFTgsMHnCGT3DFJgx3cvk9O7lyVkCyuw'
                                                '8+/XyPH3a5K/xqzAjm/fpD5v3xTbyGAFjqu9RWtqb8n/dgdr1+'
                                                '5gztx9m7J235bIv4IVuOFhC1162CKBwRYFduxoJRD35RYFTt/T'
                                                'RiDp888+QWaf1VYwzcJ5fybO++YPo0zeH0ZZi5/tp8D7vnYlkK'
                                                'Xj3RQ49/CHlHP4Q/lYRusKGc1ds5Lp8axk+WO/tz7e780xkQdp'
                                                'eeRBeXTAV4EN3+QQyKBpAeoyLQBf/zICCW0LUuDsI7V07EitbK'
                                                'k1KXAD1tpLWHNv+LICGw5dJb9DV+WCwaz1BjNnz79BJ+ffkPIH'
                                                'ZgV2c7tFrm63pPyzYAXuGnqbHg29LfuaghXYeXsLgSzysihw0/'
                                                '5Wat7fKhvmrJPEOeuoszpGOqpjqPSPPAGKOVBlBdPdbg+ko9sD'
                                                'ij+WYcw9lpGzsXsfBY4Z8LcpdsDf1tufumrLp67c0jyTbM0zsZ'
                                                '69qMDpuzbRvl2bZOcJL0094cX259Opx/l0sQvxUeCarONU++Tv'
                                                '76cYFLjjaBHdP1okh8YFaca4IM5wqKEDDjWy8mWTAifsu0Ab9l'
                                                '0Qx1pS4BV8hZbzFXm2g9Whgznm7eu0/O3rsj/XrPtyzRzzThMt'
                                                'fadJHoUHK3DDv5vp0r+bZer////zK8a1UOy4FgnLi7Oa8+JMj3'
                                                'ueFaCWHuEC5BK3WYAcZ2AfnoF9OL2vAOXObBag3EWf5sQv+tR4'
                                                'drW91q22Z8f2XJN9e661ZqGLAtcHGQnkFxd3zXJx5+gR62jBiH'
                                                'Xy6388teM/npxx8Ds6fPA72XJsogLXRRwhkNWj/DV2lD875+XR'
                                                'gLw8GTgnUF3mBHLHsNN0f9hpufPzk5+la+SY9ecIpPR9UuDQjM'
                                                'sEkvEDK3DWhEY6MqFRum0wK3DKupsE4u4VrMC2jlt0t+OW1AQ4'
                                                'Ul2Ao4T9+t8nyC7njRwwxs69LUApK0YLUM3H6QLU9P5OAUq1GS'
                                                'XNZqTZC89L+MLzFOYeYTW7R5iyPuimwI0hn5o6Qj61hrT1U+CO'
                                                'b4eR7dth8kONm6bV4P6+cxmBhE0ao8BJdVtpY91W+ewvb43/y5'
                                                'vNn2UQyPMtvura4svGD6wEErkyQIEb2svoanuZLO5tVODEUWcJ'
                                                '5F6rSTtaTdx57yLdv3dRWqewtkxhzmy/ShntV4Unm9U82cyJ/7'
                                                'lBCf+5IV330unRvXRx/8BDPD7wwD2HnxC/ITvEMGQH5Ub1M4F1'
                                                '49UrknT1CsXk3reCyeh6UoD2Td8gQLa6aOmoi6bQwCIBiv1Orf'
                                                'HfqWnfl/8TIPPk60a3yddzhjzso64PcR+634NA/ho/RIHtR79N'
                                                'DqPflsSvX1TgmS99RSATBo5X74HjOcpxD81z3CMHv/DRjC98uO'
                                                'n7bLrxfbaMWIX3Z5WBHd8vJqf3i8X/tSA1vBbEhqQa8kmqkWnv'
                                                'mxQ4tGc9gTQ0kQKHJlwhkJboO9QcfUfspnQPspvc/ZRxSjaB2D'
                                                'n/Ye3o94fJe84SnPWWyJG+CZLVN4HSPA6ZwJrlWSZAuW9vsYJp'
                                                'RUi1ADWZI6XFHEmBazeKYe2Tn0FxRICMLb3E3NKL6vu1CpBda5'
                                                '+c+NY+xhEJvdQ9oRfHxFaaYmMrrfO8Bilwx3kzgZzf667AKW3r'
                                                'aVvbevn6D0/d8ocnt3T/gdq6/yBrn5qkwA1zfqGrc36RIxH+Cp'
                                                'z4WT6BzE8I1KiEQI4+fJoWHT4tn1QZNa7KyIVt5yi37Zx8/7Ej'
                                                'g8asfZZWrH1WbAl/EEj8nMGmyDmDrWnvlBOIS8peq2PKXpPxqy'
                                                'QCaeke8QRFfvbMExJ+6ReZeekXsnv7SA4Yd+tNSdebFGM3UYCa'
                                                '16ZJy9o06tb/B7Hr/wNl9vKVfb18afm8ixIz7yJteZmt8S+z6c'
                                                '1N3RU48s5GU8ydjdb7+c4KHPr1KDJ/PUralg5TYPs1HxNInz1j'
                                                'FLjHK9vJ/pXtsuq1CRr72gSe/u0hAnnB3U+B0wxCIA27AhQ4fk'
                                                'M5rd1QLpWnh3HF6WFqf/r/CKQppD/fCumv5s2jCSQ6sAeD1rz5'
                                                'jQmsnp6XCcTtwjgrmLpm7KKHM3ZJotn7CUrz9Kd0T3/J7P/9E2'
                                                'SOvmqEnDWJ1yUu8Tp5ju0m3mO70dyAYxIZcIyyJ24UoMeXI6Xr'
                                                'ciR5Li0VoMw9mVYwxfR8JEBu+3ub7Pb3tv6n6TlNbnqOPUJ7k1'
                                                'tob2m/OkRtV4dw/DtzCGSTo4cmOXrwiqubCcTx4/EKnDZ5L6VO'
                                                '3itDyn3UtdyH7T4/QSAZAeMYsOf+QN/U/iBpf7ozqP3jBAKh9E'
                                                'EM6pb6ErmkviSdI3rzgxG9NWzpWVPo0rPWxSfbCMTu9ZRTdq+l'
                                                'BIVdzaIZV7PEzvD0E+QdNP8JWeCYKFGOiZRp2WnaZ9lppa5KMX'
                                                'ZVUtjU3VYwWeIqJCSugmoWRwtQ4WcbBMjD95gAZfh0k0yfbpRc'
                                                '2i5AdouePgVBvTb2VvuNvdn75gUTWD+9N0jj7g3izPcmU9Z7k2'
                                                'Wez0gFrjnwOYHMeG+cAqe8mUYghkwfDsj00eljT9CMsSck5NXx'
                                                'DOp4eg+BOF57kUE9aDONoc2Su2MI5+0YotHZ7xBI+hfP8Y9fPK'
                                                'cuk+zJdZK9ePT5nUDS1tqZwGrwKiGQ0vhj1pr4Y6bSjEQCCfvj'
                                                '/SeoccWfppYVf1rT9LgA2RU/ygFj7aBGqRnUSHafBQrQ4S9SJe'
                                                'OLVHLP2S1AiZ7jBKgzrEGAatKHWxvTh5smH+ihIQd6cOSA7Saw'
                                                'Nm7vr8DZq8ZSzqqx8k7v4RrRezhHv7yKFr+8SmxbAhjUPK2caF'
                                                'q55Dj4MajdbSuBuE+YwCMnTNDIsYcIJGL1GH5n9Rg1ntxG5pPb'
                                                'hCYMY9DG/31EIOuWOjOo9y53Akkf1J1Bt7htMCW6bbAmBNQTiN'
                                                'vX06xgsv34HYE8HmqQrqEGatzn9YT0dNgt9g67ye7BViPkbBt7'
                                                'S5LH3iIHW29xtPWmlNePCNC65zZL/HObKftWuABFba8QII+SXV'
                                                'bvkl2m8NWPBchuqbsJrF1Vjgqcm/QcFSY9JxEFRgaNzz9HIC7/'
                                                'F8gD/y9QM1edpsOrTkvyVH8GDfPNJ5DpTRP5laaJmvn7EQLxPu'
                                                'vJE856avan39OxT78Xpxh3BnUJWE8gpddduOy6i3b1ZurszTLT'
                                                'tRfPcu2ljQ9KTB0PSqxpT7UQSO7AwJz4gYHGxg9/JhD7ew5PUI'
                                                'v9HLLZz5F9fZKeoJiwL01gJakRo9SQm0eWFUzfHy6R1MMlFJ4Y'
                                                'I0Bd8eukM34dPU45IUCJzg+s8c4PTH0+vos14S7ZxW8LglOJV4'
                                                'i/uEK6YuEVAgl828SgudcvEEjES0E856UgjZ9dQyAOUQb+R5RB'
                                                'U4YW07ahxRK2xIdB972UTSAn73sxaP3Wn+j81p/E/d0XGTQuLY'
                                                'nWpiVJSpsrf9vmqjWfvUV1n70l32zvw6Dx57sRSNxXDwkk3lpp'
                                                'hBzvhwUEEhpfZA2LLzJ5FiQQiMuDD5+gLRfaTWC9lHZKGtJOkf'
                                                'nfw6xgClnSIKFLGijznkWA4janCJDn23sEKHfWSAFqybRwa6ZF'
                                                'fb5uJ5DcMb4MOr3uJE2rOyleI5lBO5ddJcB9Pc0E1syiCgG62m'
                                                'ph0LAFNgLZsM3CoIle7bTRq12yZ5j5xAyzrjh2g1YeuyG1jeMZ'
                                                'tPT4PgJxL7QwqPvFdhp+sV06Ngfzvc3Bagy/TSAZvxv44O8G9Z'
                                                'xQSl4TSiVhXCeB2MXsNDYu25kTbrAwaGdVK3VUtYrHUROPPmrS'
                                                'hvEXCaT0R1cGja+aRXFVs6Rj6S4rmAYcqBDnAxXk7G3m/t5mXT'
                                                'z5Bi2afEMa//Zi0Md5P9FfeT9J+N8LaebfC8Xx5OdP0HuPn9XI'
                                                'x8+yXdFfJrCm/4q75JPvCbXeRgvW2yT2toVX37bok++fBLKv0s'
                                                'KgXV3t9LirXc5/bmFQ7442mtjRJhvKgxm0zreFQKZcZAZNfvo6'
                                                'ffP0dcn5KZCtPwVq+sgq2j2ySjz/5cmgi5d8R4uWfCct7Xh+7R'
                                                'btWGKjziU2iai3MGihu42K3W2Sm27hvHSL5r7fTiARrhZ+x9Wi'
                                                'Wxa0EkjNETOfOYIzsUcTgXjeNfLYu1gP5pwnkHDrRAZ1XnGEQF'
                                                'quDWDQxD6TCKS50cKg2SE2Apl2ysKgj4+301/H28X2loXvvIXP'
                                                'H9JGIE1hwXwrLFgbPsJ97KNm6bmVGNQ7soG8IhvEIdCf/xHor6'
                                                'VP5xNI5bARDJo4Zi0ljVkrjdbLBJI52cOaO9nDlFVh4V8qLBr9'
                                                'oJ0WPWiXsPUWBi1tbqPC5japLwhm0C7XFnrk2iJxJcyg7oWN5F'
                                                'bYKPVY70Cj1pymBWtOi/NfYxnU/dRO8ji1U7KkB4OmdW43ZXZu'
                                                't4YdG00gG2alS8KsdLLttnD7bos2RrQTSNxAC3860KIxM1sJpO'
                                                'UnM9/+yazxXTcJZHOFkUHdG8+RR+M56f3uRAZ1nvAzgSRf68+g'
                                                'kUPHEUj0Z6m04LNU2bcl8AmqmTVUgKJc9ku0y36Kmmnh92fief'
                                                'q30Wb/NomfGsz/mhqsPV5tpp6vNkvnJ8S/fUKa3KOBvunRIO4P'
                                                '/XjEQz+NHZhHawbmSe4Hw1k/GK4dyasJpHJwPYHEn/qnFUx2dj'
                                                '1OQED6VzYBWuxdL0BbWqdZ41unmXpYgxl0S/cWAgk7yvzqUVbz'
                                                'vxqJ/tUoDSGBfCUkUGMqKmlpRaV0rRrLv68aq652OwjEPqI7gz'
                                                'beSTSBtWvAUAIZ4LtfnH33U/iOVAHyfitQgL681V8Tb/XnFMs4'
                                                'SrWMk6wUM/+SYtbMkpt0pOSmjNpnZFCHyHPUJ/KcJP01gb/6a4'
                                                'KG3v4vgaRddGbQlAEvUuqAF8WleTuBpL8RImlvhFBG52jJ7BxN'
                                                '+YZ0yTWkU2lxDwVOm55iAuuIpzzV/SlPDinaSeYizI/fLHz0N6'
                                                'xfm2w0apNN3LAeDMN6EPcvG639l006bBYGrfzIRlUf4ePN+Hiz'
                                                'RWeG2wjE6bKFQVu8bGTzsolrmYWfL7NoZns7gbQctDBo3Np2Wr'
                                                'u2XVbEWfjjOHy+020EMmW8hUFdD7WSy6FWyfg5mA/9HKxRF2/T'
                                                'vIu3xeGWmZ+9ZVbXG030wo0mGbydGdTTqZG8nBola4WJj64wae'
                                                'LkegJZ+VsAgzruryCQdWN9GPB+HqeJkcel97djGDRk+HayDN8u'
                                                'rvctDOrxuY3GfG6TxLsWBi1cZSOQ7BYLH2958j2ObQQSes3CL1'
                                                '+zaAbZCKSj2sK/Vls05GkbgfQ4buGnj1vUfl879dzXLus2Wxg0'
                                                '2bGdtjm2S/p0C/80HX9+cBvtG9wmcdeD+ZPrwWre1EKWTS3SNT'
                                                'GYQeu7NdP5bs0SPsDMoInN12lj83VxmUw8cDJpiPEygQyYFsSg'
                                                'qctr6PvlNRLznh8ve89P6/oone+jUmjx4gKLl6ZP/ZFA4mrcGN'
                                                'Ru5zICaXryvZHu4f2Ps9HVOJvcbbMwaOgiG01bZJOrN7A/3sDz'
                                                'mW4jkNJzFi45Z1GX520EEp9rYdC4CjzfCqyX31kY1GdaO02c1i'
                                                '5Z8zG/5mO/iG0jkJDeFgb1HNFKHiNaJfr/ghl08e/NBFK4zsxF'
                                                '68y6b+VNSl95U2IfEq9+SFpZeIVKC6/I60uMDLp4Ux0t2lQn+c'
                                                'EGBo3dU0Rr9hSJ+1ycd+dO0BWFh2hl4SFJ6hzJmzpHanbfRDrZ'
                                                'N1EqR/Xj06P6adqIIZQ6YojUYn6DhmN+z8b87riF+X7Loilv2Q'
                                                'gk9xL2n0sWzRlno5PjbJJSauHtpdgPWtupsLVdnDG/QSM+aafI'
                                                'T9rF5RMLD/zEoo6lbQTiOM7CoA3prXQ1vVWcDwdzv8PBald5m0'
                                                'A8r5l53DWzRlU2EciGr5hBH9+9Ro/uXhO7+SZ+ar5J4x3qaZ1D'
                                                'Pd7/ALz/AWr/SgX1fKVCaux8uNrOR3Mqj1Fu5TEJfX0Mv/z6GH'
                                                'UM3EYgjWGD+VrYYM0+NJVAjr1px6Bu3V4zeXd7zboZ8xs06j0b'
                                                'gXTD/AZNN9lojwnPF/MbNLW7jb7vbpMZxywMGvFTO4HM/srCoC'
                                                'EO7QSSP83CoCn92ii1X5ssuhrMoPWftRBI1zjM73HBOr3jFoF4'
                                                '9zEzaGzpdQI56U8MGv+Py5Twj8sy3CeIQetG1BCIj8mPJ5n81L'
                                                'BFKHCLSE5PLz7V00sr1+6mqrW7cWZ0Y1B77yXk4L1EjB692eTR'
                                                'W92+P2sCa8f0Kro7vUpK7ydbc+8nm9Ixv3/E/A75p43M/7TJAs'
                                                'xv0MxBNjoyyCZNauGbalFDWTv5lbXL3VS8H6mY71PaCcQlysKD'
                                                'ojC/l7eR1/I22fOMhUGjB7USiPtHwTzyo2BNaW2mba3NUvuJmU'
                                                'G7Im7Sw4ibMridGNRt5xV6YecV2TfbyKA9QusIxG6EgUGbAoqo'
                                                'MaBIaOwEBp0+8BBNG3hIKG0kg+7r/gWBPEp1YtBsu8EEkhvRQi'
                                                'DmvLE58XljjXZ/fkUgPRzexv3sbYrB/P4I8zvJ00abPW3ijPnd'
                                                'D/O77nY7gRgPWDjoAPpj2wkke7WFQW35bQRSN8bCoImprQQScy'
                                                'CYlx0I1ji5TSDmi2YGnJ+baPmxJqlbzwwaUXGN5lZcE7dZJh42'
                                                'C+fd2gt0tfaCxGsAgz6+V05/3SuXtHOTGLTQ/xiBZPcew6CJr2'
                                                '6lja9uFZfLg3jQ5UGa2RxCWc0hYuf3N/3l+7c0fhBsAmt42WEC'
                                                'aSp3kpZyJ6r/u9sTcv/SUem4dJT6YH6Ddj1lo4dP2ST5qIVBM3'
                                                'a304Hd7eKZZGHQpmdwPnqmXS6FWhjwPrYRSNjlYAadvqKFpq1o'
                                                'kRaPYG71CNakK7cI5O7TZgZ1PHCd+hy4LoYxxAFjSI2XLpH50i'
                                                'W5OySIQT3rqsm7rlq2uPjxv138NNRRaIqjSI/ycfx0+Tjt+GUX'
                                                '3f1ll3TZu/Hv9m4aqYsIxG5DL35qQy/1/qrSZPiq0uoeWUkgiU'
                                                '7fW+Odvjd5L3uVQBJTv3mCIhv+YQVT9O6rAjQF8xu0pbidQDp2'
                                                'WvjeTux/Fpy3Le0SO8/CoK6L22jw4jaxe9rCoEnPtdLG51qlbn'
                                                'Ewn1scrMkXmwmkdIWZy1Zgfltu0iPLTdlxjRi0x0dXCKR0qpHL'
                                                'phq1sXsd3eheJ03dDXyru0HdKwtpVGWh5P7Pm0FnRmcQyKNpIx'
                                                'n08f3PCWRfXyfe39dJwy4MoJkXBojLgWYCMfeOyYnvHWMM37mJ'
                                                'Zu/cJCsezsHeMYfMn/ibwLpy/CWJHX+JXMMni1v4ZMpetF2AIv'
                                                'dbGLTH/7UTyMxYC8+KxfpuxXnf2ibJHhb+jwf6k1sJpHBPMBfs'
                                                'CdaazNsEEnLWzC+dNWtIWhOB2K9h7rWGNWrvNZq395o4TTYxaH'
                                                'LKBdqWckEG/BTAoI+/LqdHX5fL/f2TGNSw6yj57Toq4fmj+a38'
                                                '0Zo5L5lAekYNYtD4QRZKGGSRxpN/Eohb1gSTY9YEa/j3GfTm9x'
                                                'myb9vAJ6hj1B8msDpYsgXIUNJh9S7pMEVsOilAxi8PCFCap6sA'
                                                '/bXRwqAx3dtpRXfs31Owf0+x6PSn2wjEsz6YQVOjW+j76BapHB'
                                                'HMoKnlt2hH+S1J/pN565+s0zddJ5AtrsT/diVdvO8SgcQ9E8Sg'
                                                'meur6cj6aln7my+Ddm62Ekha0jj+IWmc2t1Oo7+a0yTxx6EM2h'
                                                'UaTZ2h0eJ8w57737BXt2VFJpdlRdbpy8oJJDZorzUmaK+pceo0'
                                                'Aoly2voE5U4YbgXTkYgGyYxooNjXvxSgzr3zpGvvPMp/5pbkPn'
                                                'OL4vNSsB6mGGsjsf9H4v2f30YgM7tZGHTL0624v7ZKbHQwr44O'
                                                '1siyZppX1iwZS8wMmuVxk4553JSus8SgMVOu0NIpV+SIj5FB7f'
                                                'QsgSxu8uclTf7qM6eQ/OYUyoECbwZNOn6QNh8/KNLpzqBZ1xLo'
                                                'yLUEOZ/Ql0HD9/ajiL39JP63JgJpfHNHTu6bO4wefl8SyOPr88'
                                                'TuxjzKHTPMlDlmmHXDqgZJXNVApQnTBGjeg2SJfJBMoRvKBcij'
                                                'eo/Vu3qP6UCrvQJ7FxSZDAVF1mmjcN8dhf19Uyvt2NQqDj8GM+'
                                                'jj3bfpr923ZeJpM4PWfdlEIO4xzCNiWHusv0bd1l+Teh8Tg2bO'
                                                'uUBH5lyQiI0BPGdjgNomlhNI4aeTuOjTSerc6yg59Toqw1ePZt'
                                                'DYj/9Daz7+j+xud2FQxylMDlNYHIyPCcR8e5gp7PYwa/SyAwQS'
                                                'Md9VIue7UseUDlPXlA7r5r0nJWnvSWrZ84e1Y88fpsKR2VI6Mp'
                                                'u6jmcIkPPcgeI4dyAtqvxToiv/pNwlE001SyZav1w4SBMXDuJ9'
                                                'gRZKD7SI3blgBq3BPgmS9kIw73ohWGOO3yIQjwfMLz5gNSy+To'
                                                'GLr0tuH2LtQ+q48hI5rbwkpR2BDBo2sZpA7Kp8+akqXzU7WAkk'
                                                'YcY4Xj9jnHYNTCOQTuNQ/s04VHMr5+O8Nl9ijfYMWvOSmOpfEm'
                                                'v0mlLc40slekmmNXJJpqlr8GQCyfxiu2R8sZ3C1hmskesMpjhn'
                                                'vFvOlyj++CYBisuZI/E5c8hbmgXIfHNxTvzNxcbnXZzU1cWJsx'
                                                '8PIJA7M0aqbcZITv37c/r+78/FMSqY+0bhvn0C9+0TzRK3wMyf'
                                                'LDCrR5+bBOJXSgwa9fwVAlkwzMigUavPUuTqs5Ja6M+gbncKyP'
                                                'VOgTze7M1/bPZWW4+DBJL4nTt/8Z27FpatJ5CEh44M2hLjSCAt'
                                                'hpsEEnkuJyf+XI4x5vQGAomr+kDiqz4g87neJrAOtl4VV+tVqj'
                                                'z4qpQefJUSPsN+9tk3ZFtZKUCle76z1uz5zrT+q16a8FUvrmms'
                                                'NIH1+2fdFDisYRFNb1gk5yrHaV3lOA6z7iIQpxIzg3p83EQgLy'
                                                'xgBjXOvUaBc6/JYncTg7oPv0DDh1+QmcsCeNayAK2vLSMQ59cm'
                                                'cf/XJunMZVkE4jx+NIM2/esbAmn7yIVBwxcbCWTFiUcE0mjnZO'
                                                'qwc7ImG/cRSJPfMGnxG0Zp794ygTX+H1YByuTuArRNjkqKHKXK'
                                                'G4cFKPYbJwGKmPq3AEW6WUzxbhZrUuMgBd7yzGQC6dVnjNr3Gc'
                                                'NRs7fSgtlb5YcLkzTtwiTeR8doDx2TVR/6auyHvux99BRNPHpK'
                                                '/mdv0C57A/e4Wkg9rxaKbWSgAtvNqaRucypl1YUgjb0QxJ1Law'
                                                'kk6h+TFLjhs18IZNRBHwXOHnSCjg06IfeH+imwwxihPmNEpiz1'
                                                'V+Dp1nwC2TjMT4HTxuMoO17kl2X+mrXMn8Py8wkkZqdBgWNvFt'
                                                'Oam8Uyd1+AAu/7rpz2fFcu2akGBbZrKSaQ9fsDNGF/AOfuKifZ'
                                                'hY/vCFRg41+nCcTwcZACR/Y8QyB5OwM1d2cgdzxVRSCjVgQpcG'
                                                'avM3Sk1xlZ7Ie7pp+RjUVnKbDorFxvwtmlycgNQecJpP6XCQoc'
                                                '2XKY5rYclr07J+q+nRM51e4IgYxcNUndV03i0PtZNOV+lvxvio'
                                                '92TfFhRz1OIAX/8NXCf/hyVvRJOhJ9UoYU+6prsS/ndOSQdORI'
                                                '8ko/BS4MVyoOVzkyxF+BQ3bn0ZTdeTI4dpICJz/Iom0PsuTGVB'
                                                '8FNuYfJ5AfHHw1zcGXMz/E+B+elDsY34bxU37LIZB3VvlpxCo/'
                                                'jntHCWTpC/4K3JCeRyAZOf4K7HGmgEadKZCk1w0KnHu9iEA+fd'
                                                'ZX457F+Esx/tKTIiW+ChzzMIeWPsyR4bF+CuwzV8l7rornUH8F'
                                                'jtmbR8v35skcq79GWP055WwBgUx6w6A+bxjYsamInJqK5IXrBg'
                                                'X2vFRCXpdKZNMHAZr0QQB7HC+jMcfL5DbGb8H4TZFKILcxfgvG'
                                                'tzuQR90O5Imj+Cuww/kCArmM8RswflRzEYEIxgeObCjB2bhERk'
                                                'QHqHt0AIecKCPLiTL56HaAxtwOYPPaCgJxiAxU4ATP0wSSj/GB'
                                                'E+oLaF19gXw+06AJMw2c3VJEIHNuGDTihoFTrpYQyAmMn43xw0'
                                                '+V0exTZdKtJUCBzfEYP75CkjE+8GKv07TI67RU1QYqcGdxFYHY'
                                                'c5ACd0yrIZDHGB+w3pVQZWOJvLYwQMMWBvAKaxmttJbJFoy/5c'
                                                'n46yqI1lWI63uBCjzd+zTN8D4tM84GKnBOaRVJaZWsw/jA6dNr'
                                                'aM/0Gum5J0iB7axn6K+cMzLmGaMClw4+SyAvtAYosDGhgkB2Y3'
                                                'xg74mnaeLE01KD8YGTyqtoc3mVPMb4wOEzamj2jBpZg/GB0+QM'
                                                'gWzD+MBhrmcJJPV9owKHR9fR7Og6qT1pVODkH89Ryo/n5NU67L'
                                                '11gRxeWUUg/2cO0hXmIHYOqyHHsBqxYXxgY+4ZMueekR72uFvb'
                                                'G7l+yFm6NOSs9IjCr6OMXLmwjqoW1onPKaMCP04/R4/Sz0ntsy'
                                                'YF9qk5TyDDIkzqFmHiUNsFApnwp7d6/+nNdWsy6PyaDEldNUGB'
                                                '3Z8/TMOfPyzZdycosPHn/xLID/Mmatq8iRxh+pnCTT/L/LMTNe'
                                                'rsRM7UI3REsV4EY70InsSJhizaaMiSOwcnqe3gJHbYd5RAzvXz'
                                                '0bp+Pry4z3EC+S3WRztjfdj2YTbd/TBb3rrso+GXsf7mnqCc3B'
                                                'MyMMhXXYJ82eHZU9T72VMSkuKrwM6v5ZDTazly976vAj9OstJf'
                                                'SVYZ808/BbbLEwJ5epef9tjlxyF3lELvqGztxHrU6cf2TnkEwv'
                                                'cmqPneBHb45b9k/8t/xf39iQrsyj8TyG30taAvJu8IgexGH7Ah'
                                                'MIv8ArNkQcYkBc7cf5QO7z8qrzj76HRnH7Z3PE49HY/LB6t9NH'
                                                'q1D8cuyaY1S7KlDH2l6AvJP0Eg69GXgL7EPqcI5DD6gONfz6F1'
                                                'r+fIm7hrAEfhvgGyGX3AkflCEfkiC9EXjb60u0ogD9DXib7wfn'
                                                'kEYpnir8AZ3vl02Dtf3t/qr1Fb/blxSgFdnVIg0XUTFdij4AiB'
                                                'vGCZpMBNQVkEch19jegLO3iUXj94VArQV4i+tL7HaXff43IZfQ'
                                                '3oc1iWTSABDT5qaPDhwoITVFyA/RN9wD0cTxGIy7d4nt/6ctcb'
                                                'OfTwjRyxoi8HfTX/tlLlv61yH33AOQVCUiBSjj7grntKIDMf+C'
                                                'lwunMe7XHOkx/Ql4a+rgn59GhCvuSgD9g1FOe50ALJuYFf3/Bn'
                                                '41uFRG/h/jLOoMCJ7xYRyDuHJmnEoUncmXGUHmYclcD+Pgps7H'
                                                'ecAvsdl1fW4Hmu8eGUmGwC2Yc+4OlFJ2ha0QmZa/TVuf//3/uv'
                                                '7XtK1qEPOGpWDi2YlSMjO33VvdOXDV9bye9rq7w13U/Dp/uxa5'
                                                'HQC1i+PXf7KfD0X5Vm/KryC/qy0NfZP4/u98+TP9H3GH3mSfkE'
                                                'MnCbv7ps8+fwlwto9ssF0u+mvzrf9OcV4YW0PLxQ7LwMCpwdif'
                                                'U7skjaVxrUttLA0+cVE8h1q0Ebrdgv5pTQ3Dklkoe+XPR5LM+m'
                                                'McuzZdAVH3W54sM1xSeotviEVKIPuMvpFD1yOiU29AE3vJlDl9'
                                                '7MkST0Aad9YyWQU+jLQV9csdDaYpEk9AGn3cf8vK/S76GfOj/E'
                                                '+zgwjywD8+Ttqf4aPtWf433yaZ1PvixHH3DytALaNq1APkTfYv'
                                                'RlvF1IIG+iD7jlvSJqeq9ITKsMalxl4Jj3i2n5+8UySbD/ioHj'
                                                '55bQ2rklEtw9QM3dA9jzjVLyeqNUYqcEKHBGcBkdCC6TQJOvAq'
                                                '9wPkUrnU/Jmzvwvu3w5enhOQTSiT5g+2Qr9U62itsrfgqMrZZA'
                                                'OtAH3PWb0qPfVD5CXwz6trjkEchR9GWhr9Q3n0Aq0Qdc/88COv'
                                                '/PAilAXyH6GmcXUsPsQjmIvgz0Ob+P88P7RfI1+ragLyWqmEDW'
                                                'oy8BfSnvltCOd0vka/RtQV/ozFKaMrNUCtEHnG0po5OWMhmXGK'
                                                'CeiQHsPracRo0tl3MlAVpXgvPnsxV04NkKeeeBr0Y8wHlnq5WW'
                                                'b7XK5+hLQJ+5TIjKRF7/0U+Bwx4ogZxBXw36GgbhfDUoT5xC/R'
                                                'XY0T+fQEZt91dgx1cKyOmVAnFt8ldgx4hCco7A/ByP+TnewOao'
                                                'IqKoImlCH3DO/GICOYu+Onkyf0swf0vkGvoa0Rcxq5RAnKYGKH'
                                                'DlS2UEEos+YG/PcgJxKw1Q4Mw+FQTSt2egOvYM5Kj2Copur5BI'
                                                'S6ACJ+ZX0sb8SjmCPuD0h0og3v/zU2DP5/MIJAZ9wOGGfAKJRx'
                                                '/w9BkFBLIcfcChcwoJ5E30AUfPL6LF84vEP9aghljM1wXF1LSg'
                                                'WEaqQd3VwPXvlRDI+B4B6t0D5503S2nlm6XyLvoi0dcQUkYgee'
                                                'jLRZ9hXDmBfIC+6Cd9z6HvuQp5G33h6Iu2VdACW4Wkow84vqCS'
                                                '1hVUyuufBiqw29bTBLL8WKACL36viqLfq5IK9FWiLyUgn3YE5M'
                                                'sl9AEnvVpAIBXoq0Rf/NxCvE+FkoE+4NQFRbRjQZFsRB+wQ3Qx'
                                                '2UcXy0r0AXfNK6GH80rkU/TFoS/xrVICOYA+YNvkMmqbXCYOXw'
                                                'YosNmrnCxe5fJf9GWiL8uxgkDS0Jf25PndqaDIOxXSgj7gFYWV'
                                                'BLINfcCO206T07bT8gv6stAXPq+KZs6rkml3AxU4anQ1LRhdLQ'
                                                'vcgxQwH6ppXnu1jLjlr+63/Dnz3UI68m6hPOWN+elt4NIPiqj8'
                                                'gyK5gr4G9HkvLCavhcVSij5g56gSGhBVIhXoq0Rfengp7Q4vlS'
                                                '70AdtNLaPHU8rkLfSFoy90fDlNG18uXegDzu5bQcf6Vsgt9DWh'
                                                'L+JuBc29WyFeLwUq8OKiSlpUVCkX0VePvsfos9t+Wn5DXyf6jO'
                                                '9XEciX6EtEX+iYagJJRx+w0VZNZlu1zHg7SIFD9tcQyDdJQQoc'
                                                '994Zin3vjExYbVDv1QaeuaiYQAbmGtQl18A+80sIZNDTAeryNJ'
                                                '7H26WU/XapTA4N0JDQAHYMLSOH0DLZhb409IV7l9NM73IJLsN6'
                                                'W4bf74Q+J/Q8g55n0HevgkA+Rt8K9EUWV9K84kp5fi3O82sD2b'
                                                'b9NIH4Hg9Un+OB7BFVRSCl6AP2GFtNY8ZWy3X0NaLP7U41ud6p'
                                                'lk3oS0Kf64EaeuFAjVSiDzh03hkCyZQgBc7tX0sg5ntBOO/hvF'
                                                '5YSyAL0ReNvprZpQSyBX3Ari+XEcgd9NnQFzGhnEC+RF8i+rL7'
                                                'VdDJfhWyAn3AMzsqCCQbfcDhJZX0ZkmlvIu+SPQ1ppwmkJXoA3'
                                                'acX0Ug9vcCFdjRs5r6eFbL0JFB6jYyiB+jD6QIfYXo60QfSM+v'
                                                'cJ/5Cr3vn6HB75+RVvS1oC9pQC1tHlArsegDji6qJRB9wai5L+'
                                                'B+sOwsgeRPNypwTf86qutfJ/4bA9SwMYCjJ5bTgonlUos+4Bzn'
                                                'CgI5ib6TT/p+raCwXyvkMfqAp5dW0rTSStmFvjT0VX57mkCOoy'
                                                '8bfV3oe4S+UPQBd6LvPvrmoC8CffV3q+nS3WrpPjtIe8wO4sqD'
                                                'NVR1sEbM6AO2oe8u+oZrkAJHDKyl2QNr5b/oy0Sfd3EteRXXSi'
                                                'f6gLfEnKXNMWfld/R1oW/LgDr6ZkCd5K1G/2rcv7LraGZ2nbzz'
                                                'k1EjfjJyx6xz1DnrnHSzD1Tg6fcraMb9CjGHBCpg/6skS1ml3E'
                                                'RfE/qyd5ymkztOy+/o60Jf04IqurGgSr5AXyL66sZVU+24atmJ'
                                                'vlT0Zd+rpmP3qsWEPiP60jNqaHdGjXyCvjj05UadIZAI9AF7uN'
                                                'TSKJdaaUEfcBf6HqJvzFDcV4caeeZHZ+n1j87K2FeM6vkK7pcD'
                                                '62jGwDrpRB+w44k66nOiTr5CXxL69r15jkD+rDbq42ojGzrOEc'
                                                'iM340KXP/5eQLx+CxQgTN2niaQwOxABa78oIpAStBXir5sr2oC'
                                                'qUcfcHJHNYF8gr449MUdqiGQY+gDTp5/hr6Zf0a2oi8ZfZ3oAx'
                                                'nagfndgf6SWpKSWnkXfZHoc1x+lpyWn5VI9AE7uKDHpU5eXGNU'
                                                'jzW4r6OvGH2CPuDwt87R7LfOybgafD1qjNyCPpBP0ReHvoQN5w'
                                                'kka5hJgcNcLtAMlwtyOdSkDaEmnrnnAoE83RGoPToCOXV8NYEM'
                                                'HBWkLqOCOObXalr+a7UcQx/wzMM19PrhGrmHvg70RS04QyCV6A'
                                                'POHYT1ZVCtvIk+4ITSWtpQWivb0ZeCvnr0gSSjD7gUfZXoewd9'
                                                'EeiLP1lHIL+irwN9zuHnyCn8nMxFH3DKr+cIJAN9wN5fnCevL8'
                                                '7LTfQ1oa8TfffRZ/+ySYG70AcycplJ3Zehd0I9hU+oF042qTnZ'
                                                'xB7Z9QTy/wBT+0nD'
                                            ),
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [10, 340],
                                            'zlib64': (
                                                'eNodmnd4juf7/4MgiIggVhBESgRBhETkFZGdyN577+TZM8/z3K'
                                                'qkMYqmVkMVNVIfYqVWUatqNWZtqkaK2hRV5Xd9f3/kyHFcOV73'
                                                'fd3ndZ3v8/12KPMu4K9PBxDiJKdptT8bm0upC4tny9MRVPu2oe'
                                                'NkGQcUmWx8NAnc7flKpqL/okKu/hjBT1cHEfSNljHvylkdnMzA'
                                                'Lp5EjK1g/8x8GrckY+UaypkmT/450w+7ndZ4HOxO0ukqjn9STL'
                                                'uvM6j/LArTbF++vuaKpdiOtcM6cbeXAm1WGQUfcmgVk8CLC1M4'
                                                'ax7Ftcc9SVS0Ym8vFTljK/F4UMC1ealcnxfOsXXjuTNqIETY0O'
                                                'uSGqstMpbkldAjOItDO2LoFQ4xjm4UPe1KbbqWqhsKhqnLiTLn'
                                                'MS0qiQPZwaz4Ygw12X15sVGH3xEVlk+qmDKpiNVp6bTciWRpjQ'
                                                '8Z011YfFaPrELDk5ly4ueXcuKPbDIuxqF/PZknk0eQF1XJjJml'
                                                'HLIrJOFCNrMPpjLnSBymM+G8+n0y2W/G4/TJCFbPHYg8rgdxjT'
                                                'a8HGiN8w17vpvSD+WiKp7ZlHPhYhGhD3L56JbBudpE9r6cyuj8'
                                                'IL65NJHVGaPZ0NGVvPZ9qJvRGd0+K0o72xK4rBeNv8jQqSu4+F'
                                                'kJDrfy+UyVxZthKSx+EUP3Q6EMWwT/KsfxrsqNM9/3536MA8nL'
                                                '23LJsz2xf3RjfIuc4OOVvHcs48buQhyX57CrIY38X+MpfR+B+6'
                                                'gpLCnxJmDnSE55DeZZl57Y6TryLKk1//bqwqAXCmbZyNB/Wc5F'
                                                'XTHNa/N41jaThhlJ+DhEY782GHffSXz6xxjW/u8Tyvf3pUt4F7'
                                                'bVtqY+qCOWR0oKxsqRXlUwsm8px78soC4gG9veqRxpE8e012Hs'
                                                'euTPP395cbetO2Gpztzp2h1NUXvWlbUl6qKKF+EKfCdX4bS2jD'
                                                'G5RWzIziVtTjplZxPIGzwViyEQj998SJriwdYbLow+2Yv9frZs'
                                                'nWPF3C1qQmOUXNXLOOdYgU+PEiKz8tFez+RIdTKnh8fw4Y8Q3i'
                                                '/zo1ueJ23ChnFK2Q8bu65sjLdmqElDVaAK43I50xWVdN9cykoK'
                                                'yeicw2dt0uhuH8+7IRHICECTM4GWpSPY2XYQd8/04LdRHbD20V'
                                                'I0VM24TQpSdlTxalQ556yLcRqQh6k0gw1nEikLiuLe/iCcJ/ky'
                                                '+eRoRn7qSqcZffBqa8e7h1pc/1PTa6MSfYsMu68rcNhcwgWHAl'
                                                'Y0ZjGqKgWTfyxve4dh9QYCbo5jw203sl0GEHvGgavzdSw9rGFk'
                                                'vYo2bRTM/6uSJZTR8KCQx+dzKHyQxryeCUgxkQR+MYXsM97c/2'
                                                'QUbxoH8/MXPSkbpme5SYtMr6ZzZyVrhspYuLOcU2uKMV/Nw21K'
                                                'JkNPJjEjL5rsj8GkrJzEw6ixDBk4lBQfJ7J36ekwTMeZEA3ftV'
                                                'VRFStnQq9KqoJKkR8uQGfK5mV6KmExcXyIDGf01Mm8SxyPSe/O'
                                                'x3POVEwy0HBch1cHLcceq+hQrOCHvCp+OVXGs/lFJCzO5cCJdC'
                                                'b2SaSrdipjLgbSMHEi57Z58GfCECbsNuCTrWfaPi3qo2pqy5R4'
                                                'LJXRwVfcx8klXP0yH1WvLJ4cSKbJGEOLTyinPvjh3+yJcd8wnn'
                                                'WPxklbwai++RzpqWGczJc94+Kp0OdTEFdF6v/C8a7JxK+ijIXH'
                                                'lMxbkMTYQ4X4/CqjwE2LjXcOTUJD199XEfmDntXzRxPaA4rehu'
                                                'MxPJHv9mYSeLqA4AVl6D1krNnpTXx4MFkjY7EuSaPFIY8TKSXs'
                                                'L60kRKvAPX4ykQ8ieXg6CY8+2bg+Fj2tK+fufRnzFSrs0kP58D'
                                                'yONrfTeRqWj1tiKWM8q5jQU8mvbhrKTkUJbU7hZHkOdVbFtEqs'
                                                '4OHXchr+U3Hqey39GhKoic7kD00BrUPFPXldxcKtSqrXaFj9t4'
                                                '5+L1Jp2prLmQ4lRLlV0jxFwZPPRI9107Glu4EjW7JovaSQxaPL'
                                                'KdgqExqgolNvLU9i9Gy0NdLq6HAy2o7Fu8WbDbn+DFMFo+4VSU'
                                                'xsLIEDkoiemcaCz7OwGZ5HrKaQgooSCh3LMRoq+d8yGe97eNDQ'
                                                'fzyqI5O41SqQw+fDcPSOpnFMAu8OpbDhcQZVO3NYP6SAUX7F9G'
                                                'lfxuKaCnYfqqLqoJysYE8W95hIiWUyu0whrLObSqRnHManSRRN'
                                                'SSfLM5uaX/KobV1E2V8lJNWWU/p7paiDjCvnFZQmTuDpfT8yug'
                                                'Xx8Gw4jwbHUNtR9HBtKtpVmUxLy8V6dwF/Hiom3FJG3PMKegyS'
                                                'oXRU0PaCkm2TfTmyJICL00MpahVFS7d4ivcmk/IhnTE3s/HKzW'
                                                'fwF0XYlJRi+7acmClVnIqW4zNYSdvDKhw6+lM6MJhvTkWQ1imW'
                                                'Lr8nsj48Da+4LJre5bIlrJBBQSUMf1nGq9RKaiQZu0sV9Oqv4t'
                                                'YqNTs2TcEyM4ydz6JwfhaPfU0K7jsy6DYzh0FvxX3pUUzg7VIW'
                                                'FwqtXFvFoPVy3sqV9LFRs0ChoZ17CDcOR5L9NJbgLUk0dUjn2c'
                                                'csbJfm0XSpkD+PlHCisJzPDgitvCYj9EcFK4tU4plq2o3V0jQj'
                                                'nPrr0VgdT8ASmcr9skxshuUSN7eAhvpiXNPKeHtG9GZbGY3v5f'
                                                'Tap2RvhJp5mzS4/qllw86ptDoTxxhLMst/TmfjhmyWjsmnT36R'
                                                'EL9SRl4sx2ZIFcc85bh3Enq1SfTdQA13yrQcn6fD4ecYZq9IpK'
                                                'lLGoucsuh7QtzlfoWs7VfC3tNl+PpU4pkvwxSjYF17Fc+/UBN4'
                                                'X7y7r459bnoCNsVzNTSFPEsGyrgc0k7k8/CvIjz3lZITWMEXn1'
                                                'exYa6cklQl7i9VfMjSMG+VFrefdBz5Uc8ReRJ7Tqcx8WQWcSV5'
                                                'rN5YyLAVJXScUk7FqkoSfpKxYLWCtTEqDjWr2TNQS7soHXuT9S'
                                                'wJMBBgn4q9Zybhb3LIyS5gpFLM5HFllG2poNf9Kg7eltNunRK/'
                                                '8WrylmnofEmL9ERH/T09d38ysHhuOre02exok8+FCUVMcirFqa'
                                                'mcBe2q+Lq3nNdi3jsvVTHWTsPrVC3Nko4Dn+kxlhqY6mlkY4vQ'
                                                'uq9zaXxewOO/i+mxsYz4PpU0BctwHK/A7rmS2wY11Rc1FHTS8a'
                                                'uTnho7A23vGqhYaWR1qFL0tx7nA2M5+iwG70Fa+uzVM2mDkaE9'
                                                'TcLrFeFgW4njfQVuX2lQ3gzE3yYOhxMZeKcWss9si2dPZ/Z4ji'
                                                'FtQgCL1+q4/JueO9cMhO8z8sJSjccQE8e2m/gwwswHnRKrDDVu'
                                                'H8WzSnW0Xq0neLOBJ4uNWGVVY8gowymykmPXZfyvl5KkDmqhwR'
                                                'oah+s4WKEnq1sW/+jy+Dm9iMDjpfidrCCtQsatHQosTcI7p0XS'
                                                'vS6OnaNSeD8hE+sduRRsKaR2TCmjQypwnuFJY4Mvl70COTE0Al'
                                                'dzLHleyXxIzeDQU6FTMT1p9dKZnmPd+PBuDM3eEzl1J4DgF2E4'
                                                '5cdwaZUNN6tbcWVhJ1xm9WCZzwAC5w0lrXI0Nke96RWrZ/FtPf'
                                                'XifK62GLBON7LjqJHZbtV0+ryaA9ercXE30aAxsXe3iUNvTfQe'
                                                'a6ZXqRm7ZWYWiZ7c8JeW7Yt1zB2nx/q4nluxBq6cMbA82MjJ7U'
                                                'Za967mlLaa7b9WM7a/iQMlor6bTGx/bEJXqyJBrsYxVMNBey2R'
                                                'J7Ws0utY2VtPn816LngbmL3HwKPRRr4R5+/TvpqEomrk+6uxtz'
                                                'cRPErMnlgFBTlK1maoqA1RM2iQBu1jDZYNWoam6Uj4T8dvC/Uo'
                                                'B4rnfGfghZMR9y+MeP5t5KfWwhuXVxKzrwr1Wxkz+yvY46HE30'
                                                'OFZ381jf+qaTquIWCWlqCJOtbc0uFp1LPXxoDVFwYOXCmieXEJ'
                                                'VhPKaNxfjod7JZc+reL5ARl7H8pRWykJbqMi6ZWKnRfUaNdqqC'
                                                'vW8k7owVeHdYyZlcOp9XlErhPzsqaIgNASip6UcsxQzsG/Klgd'
                                                'VMXuWhnhO+TMO61g129KHh5X4btJzU/TNEz7M4VwtdD+G5kkDs'
                                                '7hdkgefSILiBhdxLG3xTz8rhQ7z3JqN1TQaFPF6WiZ8IdyOqxQ'
                                                'ULdeyYqz0cwQeeVyUyIhlSk0WqXTWp5Jxt5s7rfkcvdZPvXXCs'
                                                'lpKGZuSinGx2V8WlLBmxOVuDvKODg6iP3bQwmyE5o+OZr4qXFC'
                                                'TxO59DyZ23PS+F+rTIITsrk+I5d1C/N5OqMQm2Qx22zFvtaV8X'
                                                'c7b1a2mUTNVX+aZwfyqFso9ZoILm2O4vqhWHptTsBdk8zdHmm4'
                                                '14k581cWewfmkj0+n6ARhfSrGM6mS6MIHOSJTfgEqqb64jvcn7'
                                                'rbUzitDGHttXDa9ROzanwsKW4JhL0RM2dFKjMGZnBgWhZTljux'
                                                'LGcgNrauNC5z4z/bUWzPHMufc8bzftFEGo3Q0XsKRWeCKfAP50'
                                                'HtVPpsiCFhTTzD9El0eWHLlyJH/nfTkca9fRk305mscUPwPzGM'
                                                'iZNH0mrpGBaf8iLuhg+/H/PDd14A292DufhdGJ+9jKTdgDYkPG'
                                                '/FAKv2xHew5Z1NV7z/7UHh9T58bBjAr5kuXHgzlBL1CEacHU1P'
                                                'Wy8qBvvQ3MMP19uTse/ZDYuTHUcdO3C6gzUF7VpxuEc7Qpw78W'
                                                'KgPfcde1D1b2/anOpP2szBqFyGMnedO34dRlMVPo6JB+NJWiTm'
                                                'tpWWq48NOM7yZennoZxoiaNqczpz+2SyZlgBG56WMqNe3J8kGR'
                                                'WPxVw8qmFEOzGPhDdeaG9kQn41unIT5rSRrEsaR83fE0kcHYCx'
                                                'nfApqqkk6ePo1TOZoqAp3GgfRvu8KCoS4unzh/AunTIouJTNiP'
                                                'B8JoxI4Mn6FO4fyGCGKYdzv+cT/qyIhu2l7B5VwYn+uRxXFvCP'
                                                'uphew8X9qa8g8kQVW/fL0ZqVBKwso491JQqRM5odFJiFp9ZHqS'
                                                'n5XsO6G1p8ChW87Sq8zHzhCe5oCOyq45mYMW4dDXheN2B00nJi'
                                                'io49YXoujjYQ89GAww9G9iRWs+6O+EkReWKgEdl5Iwvk1aT8W8'
                                                '11oY1v/zARO9lM9+2ufFnshp3NSKK+Go2vzTj+KJyA/H8TmXzZ'
                                                'j5N/TWbDvUDqfw5BOyccr/FTeXYsmlP+YkZ9myBm8xhurhkHHy'
                                                'ZQjC87iuF7bQC3y4NwCQklyiaC7M1TqfSNYePmOLI7JNISlsxG'
                                                'RSoen6az4JwvKa9hq9UUAp8HcfGXUKJnRLDZNQrHzTHs6BfPda'
                                                'VgGpMJPpeK6UY63c+KPL5J9Kgql1OaYDaHhKH9GMHQb6K47xzL'
                                                'k9nx1F5LxKlbCn3GpNHWW3jkoVk8s8rB43Aul2T5LGlfiOucIk'
                                                '6ITN/yZSwJb+KFH05ipjaFIV+mceqrDGylLKbH53C2ax7f78ln'
                                                'T1Qhe5uLKJpYgnJRKUtvlKHdkITLlRTKnqZxVnjxRxeySFubw6'
                                                'HsPM5YF+C9qJDtXYv52yj81tlSDghvvze8goKqSuTTq3AMycTe'
                                                'LRvT6xysNgldjSrAQehXh4RiHu0uQdW5jNToclKnVeAq/JJ+Wx'
                                                'X/NYms9b2cpAUKFvyTR/8lIk85F3FokdCptyV8LXJMyeflyH+o'
                                                'oOR8pchGVbz4Q8bt83IMuxRkLVAyXMweyUnNzYvFOHgJH/lZGY'
                                                '/EPGj9sAK3tmJfXWWo7OQ8/0+O5y0Ft35QslBS0TJJTdlzMWfq'
                                                'NfiIrH/373K69hf51quKRmRM95FT94mCmvbCh1xTsny1ipdZat'
                                                'FzGux3afgrUUuDyPhHtDq2v9NxrlLGgCI5x4T/VI5Usv+jktCf'
                                                'VTRIaipHaRj8m4ZYhRadtfCyItsv6apHO1dPXysDjysMvEkQXt'
                                                '1FxdI/VVivVPN8qoaW5xrK52qx6q+j/3odPkP1/LxKzzCR105J'
                                                'BmzvCe8UaOTyN0Y6HRDzt0JDhp0W3/VaAsbruL9fx7d+eq7u0P'
                                                'PxEwOaOgPLXgs2wUjWJiPrrKp5HFXNtKXVHD6oxT9Gh+yijqMJ'
                                                'eraf1LNiooGHawy8bW/EudDI7r1GunepxpJRjf/aalY8qObmMB'
                                                'MTikxsKdMTd0tPu2gD7jsNuPQ2Eqwxcv+kkbR+1dwvFf26tRr9'
                                                '39V89DTRKDexv8FE4Q0Tl+3MTDxuYICbEb+ZRh5dNaIbXo27rp'
                                                'o44TdGtTLRyt+EdbUJ1TYTM1pMlPU0kxBkRiE383CpmXkKPdMv'
                                                'SLz/2UDqdYnGlApMd8XeZklUPJFY0a2KNyPNKBdKOD6TCNggY4'
                                                '7SzPIVEkufSzgNUHB5s5lz6yVevZBwt8vlwxAV/7QYcbK2cGGi'
                                                'hLpJ4tltifOvJD7k5vFin4ruQ6qpc7IQFySesUsi7a5E1t8SDt'
                                                '/nMz5a5Osc8d1jLaRGiff8KLHlnsRt8fc9d0R/XFWzS9S+XbiF'
                                                'F0kSD/ZLvGmRyHktEeFQxBGRQaY1Cx+YbcE5W6LrIbGv+xKXxN'
                                                '97jStm9g0NR0RtJqksPCwS7/xZIumBJGathCGyhLVJWlaMMZFd'
                                                'I/ZXJfHoF4mqhxI/iL8fEp4i6oTIN7kmliy1kKORWHNCwviX+I'
                                                '63EjtmpbLqhyIifpDRZZ2GvGoDKX9Vk6k2oyqz8LqNxGf5EndE'
                                                '/Q4dllh9VeKC2NtqUcue4vlPktMICCompYecBXc1pAj/V29tYt'
                                                'wMM2lqC4M6SiQXSwStkvAV+553TaJI7P294KMEv2BQOu3PFROn'
                                                'krNOZKmsEwb8+5nY+JWZ1hYLrvbiHMtFTb+T2HRUQiHO/JXgE8'
                                                'TZfS74NQ/S+ZBeQuBpOWmZWozPDMSOM3FktZmqzy1YO0r8KGpy'
                                                'ba1Ej2OS8AkSZlGbDYLfK/jRmzLwuFOC9zAF25Zo2dbDSHOkCc'
                                                'tWMysXWNjXVyJcKWreIO7FcXEuNyXaidp9EPxTwQdUCm0vLsXn'
                                                'UwU7zmjp6GNkX76JGz+Z+fprCwkDJY6ImvtskDgp6m71u8RswU'
                                                'e/lESel0RfZSF/IM7okoIqGx2zMo0MMJq496uo32oLv7hKuBrE'
                                                '926U6H1K4rTgOz6S+FbwCYK3v5PFopIypg1XctNP+N1pRv6Zb2'
                                                'LhNTP3Nljo5S6hMYm71iiR/atE/S1RN8E/FvwMwd9amo3zvTKu'
                                                'i9neVa2j02ojU9eaeHbfjM92C6GjRY9JEnlbRG81i2f8Ib7hsY'
                                                'S36I1tgk+JyKFvdjlFzUo6NehwPGyk/x4xr/82k/GjhSQvcV+m'
                                                'i/u6TfTTaQln0VcGwU8X/B+Cn/Yuh7qL5bj1U3Huqo6MO0aqmk'
                                                '1sa20h4rB4v+jHuzMl/ERPtj4rcV3wjwV/UvCd/5EoWJfLvIgK'
                                                'PMpUyG313GxdTeBtE327WLA5Kb7fX/RhreijHRJu5yTq7ohvED'
                                                'rhIHrTW/B20Xk47a/g0x9U3PfVs3pgNd/+bcKrr4XF5yycChQ1'
                                                'nyNxf6dE6HmJQNH3zYJPEnyB4Gte5RE4spIhH1VMFnl0I9XMsD'
                                                'Fj84mFV1dEv4VJNMwTNdwtkSt066XgfZ5KfC34LwR/eXE+Dssr'
                                                'cRW5aka9ng9Cc1/1MbNgjAWXPyxcnSoh/1KiWGiK6jeJ5UJXvh'
                                                'P8dcE3/d/+vQpY0L6K2V+o2faLnpWGal64mzk7ycLQ+xbGx0m4'
                                                '/J8W7pOwXBT7F7rTUehiX6Et1wUffK8v4X/mkH4sn4hU4U9jHU'
                                                'kY34ead1NwH5LDugX9iNMFU742l2l/yxnyRygtPfK5tlOBy3AD'
                                                'NroCXsqVTKkxMOqViaYbI2g9vCf7XFohy+7NrsujyE2ZQkZUPF'
                                                '8fyma1tgeNLa1ZbteXe6mjkZYFYohOYG9NDtLmUlIarIn40Yk2'
                                                'l8bQciWItxGJXA3OxXlSGeWHhYco7M+HaE/mdwvhrvA/Vm3yWP'
                                                'BLGf+6Kni3VejkkXEYQ0MJHJ+Mz495XI8qZ2itgnWvNSzvZmC1'
                                                'PoyBrinUC0/V/Xw571sULPcWuhNgYG6WODv7VNychT9JqKBmip'
                                                'K7Ri2PZQb6WqqpfSDu9akCpp2r4MI3Smr3aIlebuDi8mqaPpj4'
                                                '97mZObMn0MBIukwYzNElPVk5txM+TVYcfW5L4MteeM1w4VfdKC'
                                                'b/z5vr7YRXN0eQ3jmehXtSebIgm5nVI9hUPIj++x0J39QRv0Wt'
                                                'SDvTGc3J3syMG0KBtweds3yo/34K7ewjhceIp9o9DetX2Uz/u4'
                                                'CZiwfy+E4Ptv/agfHa1ixptMP2uz60cnbl3T8e/NhlIsMiArFa'
                                                'FcmqLgmEfpNGj6QcHMMLWTKnFO8OPbj12Abn5DaY5nahrbEv6x'
                                                '65Mn/3aMIPTBTaFcgJj6nYzE/gg206LjtzOPBNIV7nS2kqq+S3'
                                                'zja8Hm9NqzJ7MmOcUO76hP76MeyS+bJ/dhD5P00ltHMiK6pFpu'
                                                '4p7tSzQtr1KePFuko+2SBnW6+2DA7pip1rP7I+G8q6cWOx7j+J'
                                                'Q0ODhS5HkTc7kWFC8wO0uaQFFGGTU4bpdiWzr8uxzlFxfbAD3f'
                                                '/tR+nUYbi8GIv7yUks/jmYjleiSG6XxKrEDJIu5BL3VZHwNmXE'
                                                '96sizk7Bq1kqPl8pfNjp/ixwdGPRJk9G6USezA2hZ0E0bp8m0f'
                                                '5gBhWj8yi4U8SmB2Xok6sQE4m87SqiDmtoNus4fdONt8Xj6DwI'
                                                'tvwbwuvH0Vx5l0TG8ExmfZHH4jHFxI8o59mCKrrLFURcV7Hmno'
                                                'bZ9Tp2tzcQMdCLgyfh9cJQGrQxlKiSWbMwk83383gwrRhPeTm7'
                                                'j1fRbaWCJms1c9trebBDR9kAA3aRRqLU/nQcEcbIDzH89jCZm+'
                                                '8zuRmQT1ZzMVHbyrnYSobltLjz7mraD9Py6JyOE14GarONfJ4k'
                                                'fOLhMGbOiGVIbgq2BVn0Xp7P2z4lNLwsJ8JbRuIHBe0S1OK+aK'
                                                'l7ouP3qQYOKo3MKKqm72gT2zvEkXQxhVEnssh5lc/uohIme1bg'
                                                'KJdR767khVHNnjItNzvoWVdgYLfwgRHaarz8TLgKH1cWnkr4qG'
                                                'x2RhSwcmsJfdQV+KyTEZOmJH+VmleztZx10TNI+BTVUiPXZwrf'
                                                'FGHi1hMTa7Rmcr7Kpv/qArb9W8L47RXsviZD+7mStF/UHN2gpQ'
                                                'LhX7808GSDEd9F1bQkm9D+I94rmcl/YmbpmwKeBYqs9LyCxfZy'
                                                '7JqUXH6kZozwUFtS9fisNzB8n5GCNdWYCky0tDHTtVb4UjHvPG'
                                                'MsuLUEs7smnZvb4F1uEhMUAeyMjuJEegp7G3Po9aMfb/tHENco'
                                                'MqIxi863fOh6LISx1+PYOySDzn94YSv688TdaNwbUyly9idQHc'
                                                'z5uEiu/RjLmK1JZHqlMyomG6W1yO5+fozYHojv9nCWTYphYXwi'
                                                'Pm9T6eiRRZ1VHv/E+bL8fgB9Oobx6IcofnkUT/DWFD5vn8nFVz'
                                                'l8KbRkZffJVCWHcMBlKk2mOHJzk5lxMZ3ef2TjmTuBsPHgsywI'
                                                'h9kRhLWJRWmTxM3laXj+nIVNuhdbYiZR89sULt8LQzMtmtx1CW'
                                                'jTUmm9KhPfSE+e5Uwkp10A3SeE4v5qKt0nxuPeJYV8dQYfRB4f'
                                                'VuLNbyP8OTItmNysSAxHY2nZk4QH6ehf+BE+LoB+6UGsyQ9lb1'
                                                'QEOU5RNJ2K4W12PDUXE/nNI4UqWRrP60QGXp6FaU4ObfPyMAod'
                                                '//7jJBrCJ/OzPpDvZ4UwyByOa/xUTnWKYer6ON4PSSSkNpmm5l'
                                                'QxD9Mp6ZiFQ6sclt3M5bfv8pnfcRJbMv2xXzhF6HEwrt+HMfvz'
                                                'SGpDo5n6KBYXVQK6G0kcGZ7Kiqx0NmgzWaHOxic1l9Ah+bh38y'
                                                'W/Cj5sDGBucxBHz4Uyb3cELdOjeDUylg/74snzSKK+JgXPg2mc'
                                                'vpWBU0sWV8+KHL8mj896T+SIyY8vD0+m4nEg1u9CGPkgnLt7ph'
                                                'Igi6GmXTztPk1k3O/J1A9M41xoBrMTs8QMzKFkUB71/Xy4XDuJ'
                                                'U5f8edIhkFv9Q6hwDGf6s0jGbYpmd2QcWacTmD8umWYpla2N6d'
                                                'w6lInDvmw6rshFM8CbVXW+bHsA3VymkDY5mODJYewfHMmJ+1Es'
                                                'XhBLSd8E1s1K4sS1FIoc0nEamclK0cPbhL73HjCBwfUTCfzHj+'
                                                'sTA1DmBrG/NJSlCRG0GhRF73MxDCyMR3stkTnjUnisSmPCkgz6'
                                                'r8wibX4Oc5zGk/etj/hOPyJiJ+MuBXJ+fgj2n4VzOUXUoGsMtR'
                                                'vFObol0vmLZALOpzLmQzq1dln8Z53D6Z5e5K/2xtR1EgUF/tTW'
                                                'T6FmuzjHzWFEfRGJXVS06PVYxukTiLqdRK5HKm3y07lizGSpPp'
                                                'vH9uMY9t0EhvXx5RMNgg3g8oUgLl8OJe+nCCy1UcR6xjLgcLy4'
                                                'p0mEzklhw89p2N3LwPVhFg/be/LjqvF86TyRrz/zY/iJyZx4GY'
                                                'irVSjdn4Yz98BUtqhj2NUpnv9mJvLyTjLOrmkMnJrBd6lZnHo/'
                                                'hn4rvNjt4sP++ZPQ3/Qn0z4QhyEh5DmF4/s6kqZt0byPjaPoQg'
                                                'JZPsnkz0ily/Z0bH7JpP7paJKWjmONqzeLv/al8hnUuU1hRUgw'
                                                'USHiLg+LJO1JFA8XxTLNOYEZ85KQ3UrhnGM6CWMyybjlQdkCT1'
                                                'JcRQ1WTcTeCioCAjhWLGogC0WTFsHqT6KwXIohvCyegluJ+Puk'
                                                'UKZPY9qyDHqeHkVOzVguDx6Pfp0P+Z39eJw8mbqZgfy3KIQPte'
                                                'HMyprKLscYDm2N492oRM59Ke7ipVRqW2cQ/XkGY1ebcPOO5p+0'
                                                'SYzrncfl5VqOeJt5L7JJ59JetOqXxAIxH848rMaqXQjeKVZMnx'
                                                'NM/9gKkSGV+JmT+NOhN4unjGX24gLm1chRanRMS6kmqdrM/DXC'
                                                'a4tcv2KZyCiiJv5l2ThVVNK4WUNNhpGvn5u4n2rhlpPIAXn9Kc'
                                                'v3x6sqle4HSzEuULHT3cD1cBNpH8288exBQZwDrRd60bZO9Jai'
                                                'kAQbBQ4/6URGrKZ1XQB/Dh1I7sO2fPVwOPePh7OhJIfXbat4K2'
                                                'Z5U5PwU5MieD3TnZL6dnSd4cwT+wBqv0pjrrYMq1EiZ24rZMvb'
                                                'OGxdxuP40QGvWd0ZUD6BR7fieeZn4NkWFRfeluJxLJVrLf6M7T'
                                                'CA0+42JP45gozDhayZVMHlCQpqW2mYJzJX3r8GKhaIzL/XxMWv'
                                                'zViL7ND1BwvBnUVWiBKZ3SxRslLkbpFhIhVpPM7MpyS3DPkpGY'
                                                'ceq1jeSYddroFb/apZW2wiMd3MBzsLPl9ZqHhl4aTIcO5ykeEX'
                                                'i8yfHsGYMckii+dwIaOE6VRRdUDJs6Fadu/U026lEV1HE7Mdzc'
                                                'ReN/OiUuQ4kaM2fyJRnivx59gJ+Iu737w7nrKPmaSOEF5svpjq'
                                                'OgUXPTRYf9BxSmT41mI+Xz5hYsYaM7cDLVzZY6Gmq/ie+IHYdR'
                                                'zLq20B7GgXw/ethOafyce4p4xtYt6bBqrRjtFhazFQMraaOp0J'
                                                'x2IzDT0tLBV5/pv77Xlf2pdV/xvBwLZ+3OwZyd5TyXw6JZe8fS'
                                                'VcmFXFo7dKjPFaai6JPLjTyLu+JoyDzVy9Z6a/qwOW9a24nyPu'
                                                'idNQhvfzZv+jEM5nJeA5I4sri4qw3KrA+QfhK7M13HHW8/MoI8'
                                                'E7q+l+xcSug25YJvWiYZk1OX5duJozCLf/xhI1cwryHTF4LUgn'
                                                'bnIBvg7l3I+Xkyu82sYUHfPqDdSFVrN6sj8KHw/8J/Rn+JWOKD'
                                                't0YLLFiY/rR+J+2w/tuUhKC1MI3pcrtLcU6/NV1HqoSDBraX6t'
                                                '5+9TsXx6PhDTtHE42LsgdejKp8dbYzY6EjxwGEusfQjYFsqTVo'
                                                'kEtc/m9vsienlUcua5Ao8vNLgmCU9RkEjFsDAUC3zw+WkYCUN6'
                                                'csXQBrsMe/IqBzP7iSc/JAYyviCWf5wyaNxSQFNJOY6L5RxMLG'
                                                'V2uzwyD6bwX8BUGouhyX8UB4f04/muDjwY1pGqL/uxb/UoIvdB'
                                                '9zlT8Xkv9GhcHknGUkJMct4Fl+M7s4COLelkOccSaR3IKKFxCd'
                                                'aD8XzUhZK/2uD5VU8aXdxwe+jDKU0Yg75L5NvabKEXGoZcVWDX'
                                                'tZItF4pwvpBFwKEEnApDcdngjd+PQwmxdeRecmscDV35Wu+C68'
                                                '1xDHEJwtY+jm8e6rGu1HK1jwrjjioq3pRw5Itczg5LYemsSH6d'
                                                '40ds0EiGODiRsNSGd2GdMH3Xn4vLPJi22J84/2qU8w38Eqaj1l'
                                                '9Nk7ec1s/KCO5RwMKUdM4rYhg0dgqDVoxl6puBhDXb4dq5LR83'
                                                '9OIr1+Hs+Z9E42s5A78Ufr17R3KeSqjnSjy/Y6L0TDlzD0nMTb'
                                                'Iw67qa85UBTOgn4fa3gUcD0oly6kjzARNJ78uQjvZjdqsYVryV'
                                                'qH0k0XBKokkv4bPcQl2miTsrtXhsLKH5ucT8qxIx30gEvrfQP9'
                                                'lMXW8DSpF/or9JoumuhM0usYcACX+hGRefiH6xVdGszMF+n5j7'
                                                'JyV2aYT+LLQQGWbigqhdhaeo2WdhpJb2YLDQ7LmvhdaECu/+Us'
                                                '8KzypWXklkRtlQTN27E+onkdDGwt5zRrS/KGm4mM3Jld44W1uR'
                                                's96b0rkWlniKvvLWMn1PMS51ochsu/NCMRS3m4m0FnPm3Ck9HS'
                                                '9VYtctkfOun5Cp6EHkrDA8x5fQ41+JNqIG7V9KPHkoceyGhPyY'
                                                'xKm1EvWVojYOEi6SmC9fmpnoYOLUJgOFdzRUvJXh9ryIkf9IPH'
                                                'ol0fqJYG8LrT0j8XabxOvpEkPGSTjvtKD9y0zydyayFhmxitXx'
                                                'fpQSc0QZvYRezHktseaZmIN/SpguSUzYL34WiZrFSVj9aeGXCR'
                                                'YODjBTllHN7Ll6njmpCfetpP5qHpZT8dx+Ic5N7LtY7Pv8LxLN'
                                                'ayQ+lotv6SLx1GghfbaZ1PYmlMsNOBzTMLRZxjdbi6gPTaVmVj'
                                                'BnHktoxb5txL4HiH0v/VTi4RiJz7dbaBB617pesDVGirzF/Ouk'
                                                'ZGW3MsLHZNE6eCrfbPTi2P/9e5TYt7RP/F4ocS5G4vEdC81jLR'
                                                'zqKc40uhoPvR7b/1Tkd6rEqi6PQ8IvGhv8WKxyob3Yt9f/7fs7'
                                                'iewyialiZjVqLUz7zEzK+2oezTag26gRPSRDUhQR+Eh4VNtgut'
                                                'SMoLvOnumnxb3cKuE5TWKjh8TdzRaO/y7u6AITOToj8wbqhB9X'
                                                'EHCtlAu/Z3L0fCTNvb34GN6HuGxrjuyVuF4n0SLm5JlbFo6NtL'
                                                'Chi5m1AdV8m6/nssj37a5WcCw0j/APcdwZ4CeyxmAu7m3L+3e9'
                                                'OLFaYm+JOMOOEnlKC1+ZzHg/r8bTYGDJAg2lehlnPYrwnp9Czb'
                                                'wgAh64c/hoF1L7d8E53p1mi0ThSInxGy1cvWLG5nMTg0qM6Dvo'
                                                'WHVAQc7qUprmZnIsJ5KYmnF4fdtbzIa2vPAYzP3lk/g7Unz3DQ'
                                                'tNwyyY24scO07MzSg9sXtVeDZU8OJDLt0Xx1G3eBJxwwZj1bEt'
                                                'g9b3xvmLcfQqiuRVe3FuVaLnNcKP3asmpthApVrDo0gZdU8L0Y'
                                                '9O4c8uQVyPcMfbsQu3m7tgeeZO0sIgZn6VwrQGse/zZgrMJvan'
                                                'GBn0XMuxegX/lpRyf0ImHR9H4PrCE9uXvej2c1uiLw7Gw9UPhz'
                                                'bxDI/Mo85F9P5HE/FDq7nqrUe/TMVDTQXmplxcRsRRazsJ+WeD'
                                                '6JlsTUtcHxY6e2F3JZI1dzK5caMUS5WZlVer8Uk08C5Z+IeBMg'
                                                '5sKqTN+WRqFgTy+6bhyAvt8JDsmfLFCEY6BFP7LAWtugjNWhlL'
                                                'xivoeVvF7720rHygo+l6FRN0chY/VrAwQNTwqQLbQBXDjWo6L9'
                                                'QwrVpN4CINgxZpiTLr8F0sflt07IrRk9PFwMV5lSyVVVESLOO7'
                                                'bnIaLsk5W6dAF6Tk6VMlaSEy5naXU3NZzpqvFEwOVrLlmZLQr1'
                                                'Skj1STf0WObqGCjiFKlM+VWC1U0XGUmpuH1ByP1gh/q2T0CyW7'
                                                'xPousT7jsJqSGA2NFzWsTNIiX6SixEON+xE17WI1BF/S4JGsxf'
                                                'asln5BOs6J9XVi/aZYPyDWF4n1lWJ9YJOO1H56NJc1xKZocT6n'
                                                'ZXCwjhVi/bRYf/GpnoA/9DSI9XVive8POoL768mfrqdRrGf4Gl'
                                                'hXZ8Dv9wqOJ1Sy+5dK+kyo4s3aKnK6ysg0ynh3S0ZgkJyQ9XJs'
                                                '2ytYXaDAXtzxvN5K1siV3PpZie8xoW2C066r4qrgfhOc7A8ZOw'
                                                'S3S3BGwdkWKtAL7pjgbBRKAo4qmdZbxekyFa8El+ggY2q1jHuC'
                                                'cwuWM7RBzn3B6QV3S3Aj+yipUPzf/9EQ7xSce7mKL/eo/v//wf'
                                                'xZcGm3ZSwX3BLBJQrvf1twoQfFfRDcGcFZiznkL+Z/reAeCU4l'
                                                'uE+ET3NYoaZLiByb7+UcFlxUkYJtgnsnuDFKJUWCWyG4FsGF/q'
                                                'jilODMgisRXF2LGpsR4qw6KNgluD6HRO/3VbJYcMcE95/gplSo'
                                                'WCU4lw5qLgjuR8HdE1yK4JyVGgJ/0DBdcL8I7p3ghh5TktlXxb'
                                                'eCey84k+DcEtXYf6vG6081DYKrElyd4Pq80dB9vJZMlZLZgtsr'
                                                'uPeCSxI6ckFwnwquQHCzBfeP4PYK7qbgigSXIbiTGi17tmnp46'
                                                'QitVLFVsGN7KjmkuB2Ce664MJGauik0jB8h4YtglsjOAetFqvt'
                                                'WnRPtEwbKnz7PhU1ghuZpMZupfh9X0294NIEpxfcv4J7Irg0wQ'
                                                'UL7kfBHRJcXq6OtUt1zBBcnuCmCe6h4BoEd0RwwW81TJgg7rrg'
                                                '5grug+Csh+lYI7i7gvu1WUdJGz3nBeczSsNrwdnv1LBAcNME90'
                                                'BwZwUX8FS8V3B3BTfoax2dTotnCO61l/AYpXqChXZmCe664E4I'
                                                'zkOnxbFJy+eCmyU4lzyxV8GFCu6W4MaM1zNFcE71eg4f1+P6jw'
                                                'YH4WW0gssQ3AnBnRRcgeBWCq5GcE7WeioEZxJcguDeCU77Vk/z'
                                                'EAP7BbdacHbPxI+bjnWCuyW4I4JLEdw6we0Q3BzBeZzQ8z/BtX'
                                                'YVfRtnIM9sIE5wMYJrEdyAeh2tzuhYKLjbgnsiuMOCKxHcLcF5'
                                                'CU4uuIWCW7/ewNYzBv4fI/NcWQ=='
                                            ),
                                        },
                                    ],
                                },
                                {
                                    'array': '<f4',
                                    'shape': [4],
                                    'zlib64': 'eNpLS0uzP3vmjK2xsbH9rJkz7QA8uwb2',
                                },
                                {
                                    'array': '<f4',
                                    'shape': [10],
                                    'zlib64': (
                                        'eNo7IfXAjoHhgd2H+od2bKs47adxG9gzMBjYP+QwBNOsQNwhYGAPABmmC2'
                                        'c='
                                    ),
                                },
                            ],
                        },
                    },
                },
            },
        },
        'agree': True,
    },
    'generation_loop.hmr_base_while_loop': {
        'platforms': {
            'arm64': {
                'research': '7440581b174be13887db6c391e45efc9c3ee2668',
                'recorded': '2026-10-22',
                'jax': '0.6.1',
                'jaxlib': '0.6.1',
                'numpy': '1.26.4',
                'xla_flags': ['', '--xla_cpu_multi_thread_eigen=false'],
                'values': {
                    'e12185e8e33e6115473ed88c9fb1ffb8deafb07b83b7ed347f9d8f5169dc8248': {
                        'tests': ['test_generation_loop.py::test_the_loop_equals_hmrs_base_loop'],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 60,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                            'output_activation': 'sigmoid',
                                            'hidden_activation': 'tanh',
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'max_depth': 2,
                                            'variance_threshold': 0.03,
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            4,
                        ],
                        'value': {
                            'tuple': [
                                {
                                    'array': '<f4',
                                    'shape': [4],
                                    'zlib64': 'eNpjZnCwr1jkbs/2Mh6MASZaBT0=',
                                },
                                4,
                                {
                                    'array': '<u4',
                                    'shape': [2],
                                    'zlib64': 'eNqLPJl97bjTnLsAFPYE5g==',
                                },
                            ],
                        },
                    },
                },
            },
            'x86_64': {
                'research': '7440581b1',
                'recorded': '2026-10-23',
                'jax': '0.6.1',
                'jaxlib': '0.6.1',
                'numpy': '1.26.4',
                'xla_flags': ['', '--xla_cpu_multi_thread_eigen=false'],
                'values': {
                    'e12185e8e33e6115473ed88c9fb1ffb8deafb07b83b7ed347f9d8f5169dc8248': {
                        'tests': ['test_generation_loop.py::test_the_loop_equals_hmrs_base_loop'],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 60,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                            'output_activation': 'sigmoid',
                                            'hidden_activation': 'tanh',
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'max_depth': 2,
                                            'variance_threshold': 0.03,
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            4,
                        ],
                        'value': {
                            'tuple': [
                                {
                                    'array': '<f4',
                                    'shape': [4],
                                    'zlib64': 'eNpjZnCwr1jkbs/2Mh6MASZaBT0=',
                                },
                                4,
                                {
                                    'array': '<u4',
                                    'shape': [2],
                                    'zlib64': 'eNqLPJl97bjTnLsAFPYE5g==',
                                },
                            ],
                        },
                    },
                },
            },
        },
        'agree': True,
    },
    'hmr_names.nt_task_vectors': {
        'platforms': {
            'arm64': {
                'research': '7440581b174be13887db6c391e45efc9c3ee2668',
                'recorded': '2026-10-22',
                'jax': '0.6.1',
                'jaxlib': '0.6.1',
                'numpy': '1.26.4',
                'xla_flags': ['', '--xla_cpu_multi_thread_eigen=false'],
                'values': {
                    '687f5fe8c114639c11f0accc6abd3993b11a0d36cb8d9b7d42e18707a444b798': {
                        'tests': [
                            'test_hmr_names.py::TestModuleNames::test_hmr_task_vectors_equal_hmrs',
                        ],
                        'inputs': [],
                        'value': {
                            'dict': [
                                [
                                    'NT_PRESETS_3_FIBONACCI',
                                    {
                                        'dict': [
                                            [
                                                'iris',
                                                {
                                                    'array': '<f4',
                                                    'shape': [3],
                                                    'zlib64': 'eNpjYGCwZ4BiAAO9AL4=',
                                                },
                                            ],
                                            [
                                                'mnist',
                                                {
                                                    'array': '<f4',
                                                    'shape': [3],
                                                    'zlib64': 'eNozNja2P3vmjN2smZL2AB90BQc=',
                                                },
                                            ],
                                        ],
                                    },
                                ],
                                [
                                    'NT_PRESETS_4',
                                    {
                                        'dict': [
                                            [
                                                'iris',
                                                {
                                                    'array': '<f4',
                                                    'shape': [4],
                                                    'zlib64': 'eNpjYGCwZ4DjBnsAB/QBfQ==',
                                                },
                                            ],
                                            [
                                                'mnist',
                                                {
                                                    'array': '<f4',
                                                    'shape': [4],
                                                    'zlib64': 'eNqbNVPS/uyZM3bGxsb2DAwN9gA6ZwXG',
                                                },
                                            ],
                                        ],
                                    },
                                ],
                                [
                                    'NT_PRESETS_5',
                                    {
                                        'dict': [
                                            [
                                                'iris',
                                                {
                                                    'array': '<f4',
                                                    'shape': [5],
                                                    'zlib64': 'eNo7e8bH7iwaBgB1sgqw',
                                                },
                                            ],
                                            [
                                                'mnist',
                                                {
                                                    'array': '<f4',
                                                    'shape': [5],
                                                    'zlib64': 'eNqbNVPS/uyZM3YMSAAARHoELw==',
                                                },
                                            ],
                                        ],
                                    },
                                ],
                                [
                                    'NT_PRESETS_6',
                                    {
                                        'dict': [
                                            [
                                                'iris',
                                                {
                                                    'array': '<f4',
                                                    'shape': [6],
                                                    'zlib64': 'eNqrFtGzq8aCAUt4BeM=',
                                                },
                                            ],
                                            [
                                                'mnist',
                                                {
                                                    'array': '<f4',
                                                    'shape': [6],
                                                    'zlib64': 'eNpjYGCwZ4BgFAAACXIAfw==',
                                                },
                                            ],
                                        ],
                                    },
                                ],
                            ],
                        },
                    },
                },
            },
            'x86_64': {
                'research': '7440581b1',
                'recorded': '2026-10-23',
                'jax': '0.6.1',
                'jaxlib': '0.6.1',
                'numpy': '1.26.4',
                'xla_flags': ['', '--xla_cpu_multi_thread_eigen=false'],
                'values': {
                    '687f5fe8c114639c11f0accc6abd3993b11a0d36cb8d9b7d42e18707a444b798': {
                        'tests': [
                            'test_hmr_names.py::TestModuleNames::test_hmr_task_vectors_equal_hmrs',
                        ],
                        'inputs': [],
                        'value': {
                            'dict': [
                                [
                                    'NT_PRESETS_3_FIBONACCI',
                                    {
                                        'dict': [
                                            [
                                                'iris',
                                                {
                                                    'array': '<f4',
                                                    'shape': [3],
                                                    'zlib64': 'eNpjYGCwZ4BiAAO9AL4=',
                                                },
                                            ],
                                            [
                                                'mnist',
                                                {
                                                    'array': '<f4',
                                                    'shape': [3],
                                                    'zlib64': 'eNozNja2P3vmjN2smZL2AB90BQc=',
                                                },
                                            ],
                                        ],
                                    },
                                ],
                                [
                                    'NT_PRESETS_4',
                                    {
                                        'dict': [
                                            [
                                                'iris',
                                                {
                                                    'array': '<f4',
                                                    'shape': [4],
                                                    'zlib64': 'eNpjYGCwZ4DjBnsAB/QBfQ==',
                                                },
                                            ],
                                            [
                                                'mnist',
                                                {
                                                    'array': '<f4',
                                                    'shape': [4],
                                                    'zlib64': 'eNqbNVPS/uyZM3bGxsb2DAwN9gA6ZwXG',
                                                },
                                            ],
                                        ],
                                    },
                                ],
                                [
                                    'NT_PRESETS_5',
                                    {
                                        'dict': [
                                            [
                                                'iris',
                                                {
                                                    'array': '<f4',
                                                    'shape': [5],
                                                    'zlib64': 'eNo7e8bH7iwaBgB1sgqw',
                                                },
                                            ],
                                            [
                                                'mnist',
                                                {
                                                    'array': '<f4',
                                                    'shape': [5],
                                                    'zlib64': 'eNqbNVPS/uyZM3YMSAAARHoELw==',
                                                },
                                            ],
                                        ],
                                    },
                                ],
                                [
                                    'NT_PRESETS_6',
                                    {
                                        'dict': [
                                            [
                                                'iris',
                                                {
                                                    'array': '<f4',
                                                    'shape': [6],
                                                    'zlib64': 'eNqrFtGzq8aCAUt4BeM=',
                                                },
                                            ],
                                            [
                                                'mnist',
                                                {
                                                    'array': '<f4',
                                                    'shape': [6],
                                                    'zlib64': 'eNpjYGCwZ4BgFAAACXIAfw==',
                                                },
                                            ],
                                        ],
                                    },
                                ],
                            ],
                        },
                    },
                },
            },
        },
        'agree': True,
    },
    'hmr_names.num_activations': {
        'platforms': {
            'arm64': {
                'research': '7440581b174be13887db6c391e45efc9c3ee2668',
                'recorded': '2026-10-22',
                'jax': '0.6.1',
                'jaxlib': '0.6.1',
                'numpy': '1.26.4',
                'xla_flags': ['', '--xla_cpu_multi_thread_eigen=false'],
                'values': {
                    '687f5fe8c114639c11f0accc6abd3993b11a0d36cb8d9b7d42e18707a444b798': {
                        'tests': [
                            'test_hmr_names.py::TestModuleNames::test_num_activations_equals_hmrs',
                        ],
                        'inputs': [],
                        'value': 18,
                    },
                },
            },
            'x86_64': {
                'research': '7440581b1',
                'recorded': '2026-10-23',
                'jax': '0.6.1',
                'jaxlib': '0.6.1',
                'numpy': '1.26.4',
                'xla_flags': ['', '--xla_cpu_multi_thread_eigen=false'],
                'values': {
                    '687f5fe8c114639c11f0accc6abd3993b11a0d36cb8d9b7d42e18707a444b798': {
                        'tests': [
                            'test_hmr_names.py::TestModuleNames::test_num_activations_equals_hmrs',
                        ],
                        'inputs': [],
                        'value': 18,
                    },
                },
            },
        },
        'agree': True,
    },
    'hmr_names.option_b_outputs': {
        'platforms': {
            'arm64': {
                'research': '7440581b174be13887db6c391e45efc9c3ee2668',
                'recorded': '2026-10-22',
                'jax': '0.6.1',
                'jaxlib': '0.6.1',
                'numpy': '1.26.4',
                'xla_flags': ['', '--xla_cpu_multi_thread_eigen=false'],
                'values': {
                    'f5f52785907c5ea9e1e477a3a08241944c51921429907211f8f7c13ddc28d90a': {
                        'tests': [
                            (
                                'test_hmr_names.py::TestCppnOutputHelpers::'
                                'test_option_b_base_equals_hmrs'
                            ),
                        ],
                        'inputs': [4],
                        'value': {'tuple': [2, 6]},
                    },
                },
            },
            'x86_64': {
                'research': '7440581b1',
                'recorded': '2026-10-23',
                'jax': '0.6.1',
                'jaxlib': '0.6.1',
                'numpy': '1.26.4',
                'xla_flags': ['', '--xla_cpu_multi_thread_eigen=false'],
                'values': {
                    'f5f52785907c5ea9e1e477a3a08241944c51921429907211f8f7c13ddc28d90a': {
                        'tests': [
                            (
                                'test_hmr_names.py::TestCppnOutputHelpers::'
                                'test_option_b_base_equals_hmrs'
                            ),
                        ],
                        'inputs': [4],
                        'value': {'tuple': [2, 6]},
                    },
                },
            },
        },
        'agree': True,
    },
    'hmr_names.single_network_on_hmrs_arrays': {
        'platforms': {
            'arm64': {
                'research': '7440581b174be13887db6c391e45efc9c3ee2668',
                'recorded': '2026-10-22',
                'jax': '0.6.1',
                'jaxlib': '0.6.1',
                'numpy': '1.26.4',
                'xla_flags': ['', '--xla_cpu_multi_thread_eigen=false'],
                'values': {
                    '21ef8819b7ed7ae6e5ae970485f80e9b28fcdb8ab3533a8d186f0a1fbb534dd4': {
                        'tests': [
                            (
                                'test_hmr_names.py::TestSingleNetworkEvaluators::'
                                'test_equal_to_hmrs_on_hmrs_arrays'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 10,
                                        'substrate': {
                                            'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'max_depth': 3,
                                            'variance_threshold': 0.03,
                                            'neuromodulation': {
                                                'mode': 'true_neuromodulation_4nt_option_a_tanh',
                                                'modulation_strength': 5.0,
                                            },
                                        },
                                        'neat_species': {
                                            'compatibility_threshold': 2.5,
                                            'max_stagnation': 40,
                                        },
                                    },
                                },
                            },
                            {
                                'problem': '_TwoInputXOR',
                                'get_data': [
                                    [[0.0, 0.0], [0.0]],
                                    [[0.0, 1.0], [1.0]],
                                    [[1.0, 0.0], [1.0]],
                                    [[1.0, 1.0], [0.0]],
                                ],
                                'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                                'input_shape': [2],
                                'inputs': [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
                                'jitable': True,
                                'output_coords': [[0.0, 1.0]],
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                            },
                            42,
                            [0, 4, 9],
                            '<f4[4]',
                            '<f4[4, 2]',
                            '<f4[4, 1]',
                        ],
                        'value': {
                            'dict': [
                                [
                                    'W1',
                                    {
                                        'array': '<f4',
                                        'shape': [3, 2, 340],
                                        'zlib64': (
                                            'eNrt1M8rgwEcx3F/wXMkxUFLy0FL1A6y55OkOTxWflwMscOs0FJKHE'
                                            'TDgWgXWZliK482Bxaz+fHsu112WHKR/Dg5KMtl0g7kwN/wXPd5Xd7/'
                                            'wdu5kkLQ5YYtvIremmZJlg2M71zg0h5BsV5DbCQNq0fHVsiH/HyrY+'
                                            'wqjmjnEsq7g6q1qUN6guv46btW3xvqZGYyJFVERBWq/dbA6Ns+AoUY'
                                            'dMWm3rmzuPcb8FUn4Xrcg2LJYLs2hXgiCvvrALRCGs6DI8x6/BjWzh'
                                            '3P+WOEWwLwWZbVl3Abf0pEFcs2F5HNk4Q8KbrEv8/4QyIik6aLOVko'
                                            '5cT78d/PHH9KRGTSg/dQGqVbvrpOpTC1xp8SEZn0258VY/FGJkpZGd'
                                            'rI8KdERCb9AZZic0U='
                                        ),
                                    },
                                ],
                                [
                                    'W2',
                                    {
                                        'array': '<f4',
                                        'shape': [3, 340, 1],
                                        'zlib64': (
                                            'eNrbaTPhwMRfqQfWh9k5HL/V5PDXbNUBrQ2LD/gXzT7gKzbxQE188g'
                                            'EHBpsDV9/yHsi6unm/zVQm+/x7G+0LuXgd1ltZO9S9bHa4/WCCw3L3'
                                            '2Q4hfYsdGEbBKBgFo2CEAkWD/QdunDt04F3C/gNPLhw6MBoio2AUjI'
                                            'JRQB7I2bHtwEnWAwdsNLYdWPp4/2h5OgpGwSgYBWQCALz+PII='
                                        ),
                                    },
                                ],
                                [
                                    'bg',
                                    {
                                        'array': '<f4',
                                        'shape': [3, 340],
                                        'zlib64': (
                                            'eNoV1nlY1VUawHFTTERFJRcCzCVFJULcFdGvICBuiIQIyHpBdu6+Af'
                                            'feg0pmWmbmPo4ZKpFjZo4L5TKmlGuIZNogmvtouTBomubYvP1xn/s8'
                                            '7+/5nN857znve34FY7P5dX5vJvvp2V05ke11+ayY8hY7H7xJWWgbPM'
                                            'J0HDaksv3ueAjswkc6E6+tmkfj/mn8q7EfkX+3MuxZIZVRc+jbeQTT'
                                            'hhdx6O0sduycQyv/aOp3j+BpfS8897kR/E03Es5oOTEwl5fXpbB+YQ'
                                            'yOJaGsu+iPK9eTrYM7cMPbgDWtgOwXGbwUG0/LuUmcdQ7h4r2ezDa8'
                                            'xAFvExnDiwm+k83FZUk0LZvK8arRXB/SF6a5433BTKudOtZo8ugelc'
                                            'aRvbF4T4XYHgHkPOjK4rlWtJcMDDYXEuPUUB6TwOH0KDa+P4xF6b60'
                                            'bLcxodaEa6CWSeNzqEyey63r01m7KISUBf1ZfdaOrsjC/bf1vPVBPi'
                                            'evppNyPg774zDuh72JJqaYirfzOeI5j/hz6Sz5JomltXE46qfy6Ocw'
                                            '0p+Mxm/gm1S+1xd9XHfidrjzsK8bfS51YfOkXhhXaWl2L+Tc+Ryi72'
                                            'TyZ0AKDYtnc+DhDIZmRfL3C+OoTBnKNg9/NO18WFHRCdvBVuR36kjE'
                                            '37zZcUyHzVzE+YV5eF3JYqEpjSeDE1ndEku3I9EMXgV/GEfyTBtA/W'
                                            'evcTvWizkb2nJhRDtmXX2F0bf0RJ0o5nmPAi59NY8eGzKoqU4m6/u3'
                                            'yH8+jcAhk1iTN5bwfUGcHvU6zZ174mnzoDmhNX94d6Zfi4F33XXYPy'
                                            'zkvC2Xuq0amtumUl2RQIjXTLpsjSIwdDzzrw5j6z8GUnjIl85TO7Nr'
                                            'cWvWR3rgumske7ge9aiIIN98TnyYzYrwdDq+mkRtmzjKH0+h5u5Env'
                                            '46ihttA5mS1IfrXbthyWlHVUFbYs6baJlqIDRMi9/WAoZl5rAtPZPk'
                                            'pXMpOBuP5vUZuEoiCP4xhIRJwXx5qT9DT3lzaEJHvlzaivd2momONd'
                                            'Jo19HQo4iQ7nlMT8vC2pRKbdkczrwRy4urk3n+twm8ohlBmymDOW3s'
                                            'hbtnV7a/5cYghwVthInSDXoWGIrp9kU+m5hHSqcMFrZJpluXt3g2YB'
                                            'o6wrFkjOHW2jfZ17YfN+q78+OQ9riFWMkZZGbk5wYS92p5NKSQBrdc'
                                            '/HprcOSnsK1+NgWRMdw8FEmf8aGEnRpK0Hx/OlT4MKqtJ89+seL/Pz'
                                            'Pe243Yb+nwXFeE1xd5nPPKZuOONIZoE3FMnMXvr06h1RMIvzySbdcC'
                                            'SO/fm1n1XjR+YGPtUQtB6020aWPgg1+LWUMB1Xfmce+HDObdSWZZz3'
                                            'hU7HQi3p9Eev1Ybg8cwpMdr/Pt+z0pGGxng8OKzm6mUycjWwbpWLmv'
                                            'kNNbcnE2agiYlMqgUwlUaGaS/mcUiZvG80vMcAb0HURiiB/pNXbaD7'
                                            'ZRP9nC5rYmtLP0jPEuRhuZj/5oNjZHOg/nJjElNo4X06cydEYYz2aP'
                                            'xmEP5M+GPhSNL6H6hI1R7a0cv2eifa6BPRotx04X0PxBDvGrMzl8ci'
                                            '7jfGbT1TqDYecjqB43joZdwfwnfgBjviohJN1O+UEr5u/MLC4wErxW'
                                            'R/tQOY9heTR+mIXJO437h+ewuzSWWyHRnH4xgYl1Iyg9OJhlBjsLzi'
                                            'mef1tCUpNiR2IRjhsOxryrKLqv2PiKlidBTowrFT2aFeHbdCw1Otmw'
                                            'UbH2v0r22cBPXzhp+FTxqEUR6JnJiwEmnt4qxc/NxblxCvNuRfM1xQ'
                                            '+PFC8yNbQcNNFtQBkr/FzERcoYNYrkG4q03xRen2Uxeqac6Ywymoa7'
                                            'SIqR9+xX7LypuCbPv76ejVejmZq1Zbw81UVLguLOIcWTW4qMx4ppXj'
                                            'nUplkoryujf7qLPumKrkdkXrcVF+S598hcllyyUPuSg/EmF7/kyDu/'
                                            'VSTcUUx5oiiZnsfWBCsbhzlIXyTz0yruHlNof1HskedHEvOJOWklIN'
                                            'PBmrUuMiyKLScVpb/KOn5X7H03iU/25DBtj47OVRY0ZSUk/lpGqtmJ'
                                            'qcDF4zaKhVmK65K/I0cVlY2KczK3SsllTxn//pxkwiNzSeyuZ/kNC4'
                                            'mbS1jv5mBkhZNks4t+Hoo5uYrITxShMu9lFxU5Mvfn4mPEL+83l3YN'
                                            'ucSZ9FT1tZJ2soSJvRxs/8hJa5cL/y6yj4WS082Kz79TGGTPH4mPl7'
                                            '17R/yWO3N5MTePiDN6klOtlDaXMGukg9pKJ9p3XLj1UOyXnFzcquh+'
                                            'XBF3SeGU3GwTf0D80M9TCL6ex9jBBnatsbKreyl10x24vnSyabmLg7'
                                            '6KqUbJebWcixOyL5cVL0vuXoh/ID68WHpvbj4h8w3srbfiEVLKwSwH'
                                            'l/7lZN06F/F9FbWS85BtilOS91Y/K5aIn/lQ0U/yf3lwGvo7skcXDG'
                                            'jdbbybWkrvUgc3v5f8Vbo45q/wL5H1ble8elpxRrzHXcXH4uPFd7me'
                                            'xqq8AsrfMHJ5go1h5aU8/cDByotObm5z4R2osDjkrO1QpH+vWH9F8i'
                                            'b+nvgK8VfWptPnZgFNTiNdzTY6VJYyY6uD5ttOQv7pInqo1JhSaHZK'
                                            'bdXJGFdlDfcUY6U2dolPnJaBb3ohOXVGOlTb6HG0lNe+dvD7b05S9r'
                                            'tIGCXnZYGc111ST2cUfaSuSsQvEH9VfPmzDFacLySgl4mGRhsp10vR'
                                            '1jnY1drFtKPyfqnHG28rJkhNtj6raBJ/T/wp8Z2eKrKrMlk2rYjgAh'
                                            'P6jnYuty4j4poD384u3E/J+idKHS6WOtqrCGhQrLgua5A+4SW1OVa8'
                                            '50wNfoeKmL/HxO1QO5V9y/j4NwejfF2sbnBxOkJyvlRxe58i+gdFhN'
                                            'R9nfgE8dniFz3SEBFUzIA/TYQV2dlOGRXuTtwHunj0b6m3KYrqZZLD'
                                            'rxSZ0rceig95oFgn/n3xP63OwmtDMf6TzVSst/MipYxHPk6WD3PR/6'
                                            'qLxhkK/YeKXOkpph8VG6SvbBbfJH73X/Mflc3ydlqWvG9m1zE7m0rK'
                                            'aAl0cna8i0G3XYyOU/T/qxceVLjOy/yl73hIX/SV3tIkfs1oAz2vmf'
                                            'jZ28qmOzZ2N2kZY9Oz+p6BleEmRjww0FHu2zdK5b5ZKX2qzEzEKgv9'
                                            'Vlnl+85G6Gr5d9moibWT0bmE88uKWavTkhelY/Mreqov6Dm7woAt0s'
                                            'iDB0aSJ+t4r5ueRT/p2fKRgbAoIzubjUR/ZGJukJmsf+uxrTTgMdmI'
                                            '8b9GWq004THEzOUjZk7MtPCDxIe2GKmReI3EK46ayYu1sOO8hU3SB/'
                                            'WrTOQFmwmsNfPyLAtRFywEz7HS8ayVXpE2GiReJfHLEj8s8VUS3yTx'
                                            'vrttJPWyY/nJwqxEK30arLweZWOjxM9IvGW+nfCrdqolXiVx3z02ol'
                                            '6zk7XAzg6Jp4SWULWihAk/F3EivpivjhXjM0buoq1aMrrqSC3V8eyK'
                                            'johIPZM/1dOxnYHKbANdDhvQvCr3uN7IlW+NhB4vZr04a5WWRnE/it'
                                            'Nd1bFXXI24UnEd5xmwizsuzt1gJPw7I+WvmjgjNfBI3GwvHTPKdNwU'
                                            'FxClZ1C1ntvi7OKuiAvyMVIkbou4K+ICC018+LUJH3cz34pLvqZjg7'
                                            'g14ma7G7gmLvobOQ/i6sW5HTMy0cfEYnF3xZnEDYw347XRTOfJetw/'
                                            '03NUXEyO9FRxz8QNMxrJEbdR3C1x0ftNnBbnFJcnbsUtM+5vyl61N1'
                                            'AjzueIgQxfI6vFHRf3P3GTikx8Iq5/ezPnxO0Xd1Ncorg+RgsReyws'
                                            'EHdM3DNxg44bSfU18bG45+Ic4gJmm+nysZlR/zFTLU4rboU4nycWuo'
                                            '22kmoyskTcAXHPxSUcMHFO3Hxx2eKWiHsq7oC4y+JyxKWIO2Wx8vUu'
                                            'Kz5+JpKKTXwpLsjDzAVxNeKaxE0JstDBZOGNvRZ2itsizstqpdU/rd'
                                            'juWykfZKOjfGssEheUYMZzk/zfNrNeXLI4u7g/xN0XlywuStx+cUfE'
                                            'aTJtbF1ro0KcRly5uF/EVYurFRf1u4UxY+Ssi3tP3AtxbvIduEXcDX'
                                            'Hf19nIa2PnB3EhQyw8Ftdln4Xl4srF3RF3Vlz4A3mvuBvi+q2T++KM'
                                            'jCHu8Sg7Hvl2oswW0sQ1iTspLthmpcduK++Ie1dcf43MVVy0uCviho'
                                            '22M0mcn/S+oyfs+D+14DXWilVciriT4k6Jyxa3SdwicX5udorEOcTF'
                                            'i3smzvq7nboBJRwSVynOs1l+ATaqxF0RVysuUVyVuL3ilooLPmnnH+'
                                            'Ja+0vdxpWgcZYQJy5W3C1xvdfbeKnexkpx18TdF3dUXJ64K+JGidOL'
                                            'Wynu009L+LK+hP8DCiuvEA=='
                                        ),
                                    },
                                ],
                                [
                                    'rd',
                                    {
                                        'array': '<f4',
                                        'shape': [3, 340, 4],
                                        'zlib64': (
                                            'eNoV131czff/x/F+hq9tRsjEsEaIJUkXpzrn9aozLBaaMc1CCDGjmd'
                                            'HMrBFrmDULjUZiNIuFRumc1ytdX6kkNGJJSB1bs2bN1/b9Pfvj/kd0'
                                            'e58e5/P+vC9i4kbwJ3Ej1HH4FzRg+BfS9PxaAvH8ZIMYPtlAadM8+f'
                                            'g0T3W9cZZAxn/zK4HYnU7xh+zVltH8vmW0Jm45SHu3HJTea3szaIhz'
                                            'fwp27i8GSieQmh09pXZHT4rpU2kB4/CpDeIytYFOvDeOQaN/yaB1v2'
                                            'RI/MYX+auNL2r4+iUUsX6JZHS9SyBO1bEWu+pYY+0pO6o5ZSeT689I'
                                            'cP0Z6rzXh5/a66Ous/LIZVaeTO8zkkEjR+ykZSN2yt0//8OgMT7ZJr'
                                            'AG9I4jkAifRRLps4hCU/wYNLZ/BW3sXyFN3d0ZNCP4ezoT/L0EWPox'
                                            'aIybgUCi1xbQmrUFEh8jVjB1nuTOoJ0dj1Enx2MScm44g3b/YiuBJE'
                                            'xw5K8nOGrrAn9qW+AvKyydGNRu3EZTu8dGa9yxItpyrEhC/DOsYCqe'
                                            'uoxA7L7e3IECqp43gXXj5hsSu/kG1Sc6C1Ddj0cEyPKPB2f/46FtIa'
                                            'cJ5NTlUQyaeH8P7b2/R6ZGD2bQnO7zydJ9vtg3PcugTsZ7JrBG7b5F'
                                            'IAFv3LCAcf2l3QRiv2eidNszkZxi7pgcY+5YIzpbBShn+CkrmIwDiw'
                                            'WoIdiLQesnZ9PNydkyeJYbg5ZvO0Sl2w6JS9FQHlk0VNOcPiWQ2Ct9'
                                            'ePOVPhoy/CUKHv6SRH/cTiA5I/80gsUp4ycanPGTBMx6qgNV+Zs6SD'
                                            'rtkzTaRzmpBktMqsFoeaZJgGIn+DCovVsu9XbLlfoZYxnUdvg4NR8+'
                                            'LoUhLgxafOJLKj/xpXgeG8BexwaoU+kkci6dJEmvdOV9r3TV1m2pJr'
                                            'DGx18kEPuFm61gevLzpwQS//N7HSgmNtYE1t56URz0IkW3GXhtm0Fz'
                                            'BheTDC6W4IZx/FoD5vPCM7Rm4RlJd3flk+6umrYgiY4tSJKQzU78+m'
                                            'YnrS94l0DiH/XgLx/10Nq2zgTSFNZMIHZNP2WDf3RcCq2JSxH3pNHi'
                                            'kjSaUrf3oLTtPcTvVLoYT6VTaoQfH43w0yfGC/TYeEGe9PTm//b01h'
                                            'kPrQR4h8ewzydjdPuKowQyKm4Ygyaf2UIgjdHPM6jjD+7k8IO7ODna'
                                            '8UuOdmp34TVTa/lr1vqQHAIJDbtiDQ+7YmrNmEUgp1J2SHrKDgo/4M'
                                            '/zDvirw3cXqfd3F2WAzYdBA6LzCSTudQ/+7HUPre90km52OilBVSMZ'
                                            'NCg+gUBcHwxk0PgvZ1HCl7OkNelpBk3efsEEVrvu1wkkudcYK5jixu'
                                            '0gENd9oeKyL5Ri0owMWrXyMlWsvCyFG30Z1NavlFr7lUpzlieD1gdn'
                                            '0c3gLOkcPZpBg1OSCaTHB0MYNLPpQwLJv96LQYv396X8/X2lm+Uhgc'
                                            'R4f2QEy/La4wSSUO8g8fUO1BY1lv+MGqsxnU/Qhs4nJPLd0QzauiGZ'
                                            'bBuSJfIjFwb19I0nkPaIIfx3xBCN2vQhgcQ2D+BNzQM0Z9prlD/tNU'
                                            'nP68Un83ppQJ++BJKa0pWPpnTV9P1ppsz9aVb3NQ/JYw3+nj9XGu0e'
                                            'rbQkJ1VTSlK1tM6NsYIp+sPjBJJwqq8knupLdZ6bOkjMwJUdqFvnkR'
                                            '3k1NHDknH0MKWf2GoCa/3qKmlcXUXJHp2tYApZ+osAxQ/pKkC1H2QI'
                                            'kHHwYjEMXkxBBz+T8Qc/o3W5HgyatvkUgUx+bgyDOv72HYFYPhrF1o'
                                            '9GqX3ubgKJuO7MoHU7YglkcusgBo0qnEOrC+eI5wd92euDvurywygC'
                                            'KR70LJcMelZjZt80Rc2+ae1m+IdAnIJdTWCNnvELgUTZdbHG2HUxRX'
                                            'pZCKR27H1r/dj7ptorXxNI6o3gDhQQOYWMkVMkLQH/lvA1ua+/agLr'
                                            '3N45Et47h+zWWv0hu9OEZuk8oZncJ+RYwWR+vkDGP19AEWvMErnGTB'
                                            'VFiVJVlEjjH49jUJczZwhkzRfuDJr/v2NU+L9j4jDTlfvMdFXD9STy'
                                            'vp4kLkNG8IghIzT1+jY6fH2bRFqcGDRx0EpKGrRSHt9yZFDH7kwgVU'
                                            'E9GTSorCuFlHWVpJSnGDTk/jYTWDtPbiEQu+Mzs+1OzPTv5lZKz7iV'
                                            'SuqKVGv6ilTTwKJDBNLaPKoDuWRGdZCpSzdK8NKNFPJ5rw4yefaPEj'
                                            'z7Rwp/d5wp6t1x1tJvrkn5N9coJmyxFUwJsy8LUFDhIAFKrfhegEIG'
                                            'ePG0AV7que8ceew7JzNax/IbrWM148yPBBL6jBu/9Yyb1m9NoYatKZ'
                                            'JpHMlZxpEacWInLTixU8qnD+Wy6UM1PPwTCgv/RKb2GMigOabplG+a'
                                            'LnYufRg0edQgSho1SGKmd2PQeB81gTXk00cEYpdTZnTKKbM0zb9Kd+'
                                            'dflRjrdCuYop6cIpCQqKclNOppco/8nDwiP5fVHy2QqI8WUPsW7w7i'
                                            'sHB/B0ruddQEVrfoUnGNLqX6s7ssOWd3GQ+73ZFUtzsUt6rVGrOq1f'
                                            'T423Py5NtzFOXszSudvdXiKwQyxDCOQbf3z6Ad/TNk/P4x/Mr+Mep5'
                                            '9Sh5Xz0qYW4v89tuL2vkc99QxHPfSMDZYRx4dpiGTfmMZk/5TBb8O5'
                                            'hBwydFEEimPs9n9XkNmeZBIKk/duejP3bX1v+0muy6tVpbP7Hj3z6x'
                                            '04D3Zpjc35thrb14m0DqH6Rbch6kG8ujzhNIvOtFK5i6lX5DXUu/kd'
                                            'A3AjpQ++ezO0jw+C86UFX0Hyawpv6RJUABI1KMTiNSLAu+bJSILxsp'
                                            '/n+HrGBqG1wmQKEDfHjWAB9dv/s8Re8+L22LPRk0bXQmnRidKeF9xv'
                                            'K8PmPVtTCN3ArTxOGxK/d97Kp2/9tP//67X8IejOA5D0bo+uU7CGTi'
                                            'mZcYNMr3A1ru+4GMdxjAExwGaCpPJJDxYs+viL1GufYkkFGLuzBoVd'
                                            '8DJrCWx/xGIHaZNn+7s7Zs+5cqCcTQ6Sure6evTGFrvieQxs8GS/1n'
                                            'g6l12ccdxME1Whxdoyku8MUOkvrS9x3Iadtyk/u25dY1Z2ok+kwNhf'
                                            'iNtob7jTZ5G6+Lp/E6xf/jw1/946O1P+XTlZ/yJXODF2dt8FL39mzy'
                                            'aM+Wrss9GDR2zkkCyU9xY9CIzw/Tgs8PS0z7SP6kfaTW99lFIHW+zg'
                                            'xq2b6RQFpHDuLfRg7S+qfeoqan3pLkjQ58cKODhvYbRjP6DROnv55m'
                                            '0NYPL5rAGvH+YwKpd+ppqnLqaU3oXkcgdq8Ms4LJsCiTvBdlSvFnj6'
                                            '1Vnz02LY/9kkCK62d2oNp7r3SQAf0TZWD/RIp6I88EVvsV+QJkd3Jb'
                                            'NviHNDTLjIZm2l1pYNDuSUXUI6lIDJ96s++n3vrkY6V/P1aJODaOFx'
                                            '0bp/kpPxGI/cvu3PNld0099T2B1Fx/mS9ff1nrJ+2jhkn75Nr24Qxa'
                                            '/PhzKnz8udzs7MSg9d8to9rvlonT/zkyaGaJL2WU+Eq9dw/+xbuHhv'
                                            '9oRyDFszpx0axOGvDrRyb3Xz+ydvdqIpAYOw8LGJO+KiSQxLAsa3JY'
                                            'lin+8AECOXx5nCRfHkehsUs6SMXvW6Tq9y1kd+g/1PnQfySq9HQHso'
                                            't8xmQf+YzVvfimuBbfpFNf+TJo5yml1GlKqThE+XCfKB9NzsullLxc'
                                            '2VLlyYD1MYsM+7Jk+a6x/O4urI8xJygz5oSE7h3Ns/aO1mKHg1TocF'
                                            'ByTrnw+VMu6vggnp5/EC/23w/hnt8P0Uj3jwikG73AT9MLGjd5CoEc'
                                            'GdmbQeMW9KPtC/qJ6+D/MGiM7aQp2nbSar//DwKxq/3SWH/1S0vk35'
                                            'do8d+XJPno+1YwNT37I4EE+PSWIJ/e1NYe2wHvxjsdKPq6awc5fCtF'
                                            'km+lUJVlpwmsnfIrpHN+BTlP8OOhE/w0sryclpWXS3mwgS8EGzT6Qg'
                                            'Gtu4A9s96LQZtes9Ld16zi0ujBIxs91PPvUwRS4T2GQS3rcJdZd0Tq'
                                            'T4xiULtle6jzsj0SOmAY1pdhmjprMx2btVmSTIMZND1lHoGUX+rLZZ'
                                            'f6aubPrpTxs6vkxD/L5+Of1eRFt01gtQT/SyB2Vh+TvdXHWpVUTyAB'
                                            'R1otYEw+baX9p3FOnttgDZjbYLLF7SKQ2qmTOlC68+uU5vy6tJV+1Y'
                                            'GqVvxiAuvOWSIJs4S6tfgxaOagSjo3qFKqHX0ZNPVAMYGEXvfmWde9'
                                            'dXtWDm3NypGI/p68sL+nJr50lkAqTrszaM6nPxBISKwrvx7rqrVrv6'
                                            'Vra7/FuXsEzwoboVF7thNI61Mv8W9PvaTRdVEEcti3P3/n21+TBwRS'
                                            '6oBAST3dk78/3VOjxzxNq8c8LTK0M4NWhX9lqg3/yhqywUZTN9jErv'
                                            '3LLDu7eN/8PWUEElVw0Aqmbou/I5CMxSMkffEISp65uoNE7Y/pQFEr'
                                            '+9LqlX3lyp7jUrsHZ+FYf3431l/dx18kkIjLvrzosq+2mcrooalMwv'
                                            'J8+O08H81oyqNTTXli7+PFPX28tPv/zhGI9PFg0NQx6XRkTLo88XLj'
                                            'f7zctHHgIWoYeEjWrxrJH68aqeVzviaQtoSh/EfCUI3/NIZ2fBojG+'
                                            'YNZFCX0TMIxGNbHwYtfvdFAqkq7cYXS7tp6/J8U/3yfGv3lL8IJMB8'
                                            '22hnvm1xPFtLA87WSk75BCuY1n+WQSDtj7uI3X+7UNWwbVQxbJtENM'
                                            '2VyKa51D7Dr4Ms2Jck4fuSaHxvI0/obdQa8yUCydzkx6BPfrhAdmkX'
                                            'pPAbA4Mm3Cik3TcKxY69GTR2t9DG3SK1oeMYtPvWDHpma4YkFI5hUM'
                                            '95qQTSNfxlBh148hsacPIbiWwbxkvbhmlwrzgC2TLpRQZNHbAYz3+x'
                                            'ZLzQj0+/0E/tVnnSk/c8JeHF5xg05nGbKe5xm7Wu6//x9a7/p/Wn5p'
                                            'lyTs2z5ve+QyDhd/ZYYu7sMcbcyyWQ1CWl1vQlpSaXD/YRSGORUeqL'
                                            'jBTlP7eDXPtwm9R+uI1W7zIy6OqBlwmkydmf7zljPjyoJLcHlfJoni'
                                            '+DGpJKyDupRBo8fBg01XaeDtvOi9tmTwYN2J1JIEmGsQyaM/445Y4/'
                                            'Lu0jRjPo4T0HKGXPAWkY5sKgyUO/pJSh2Lf++xIX/fclrXqwhmoerJ'
                                            'GYpQP4k6UDNNEcRCDG4b3YNLyXRsX2IpBOjV0Y1P6Dw6ZuHxy21hxv'
                                            'JRC7PcONYDFsqiLvTVUSsmyrNWDZVlP3Hj8QyOFuL0hqtxfIYUNMB8'
                                            'nZvboDlfcwcVkPk2ZeukJnLl2R1ix//i3LXw93rqbkztXi0s+PR/bz'
                                            '01CXcgIxOxgYtK5LAYEc2evFoBFvWWjBWxZZvsODQYMunSSQ4RfcGD'
                                            'Sp33cEkuk6is+6jtIm2UXNskuc1jnzS+uc1eGtTfT8W5vkWvQgBnVq'
                                            'mk0gTrcdGDRm6giKnTpCQpY/w9OWP6Ppu6+YwOrwxX8JJPz0ABNYg2'
                                            'bfIJDWaEer3YeOpqqmLAKxP/aHFUxh/l8RyJNpOJuHTKdun5gY1P6d'
                                            'WgLJfcXIoDP+vEQgcTl+/FmOn8ZsriCQ7Y0G/qLRoDPuFlHo3SLpus'
                                            '+bQcfXKgXVqriWjuPRpeM02P4MTbY/I97T3RnUyecYDfTBfao77lPd'
                                            'XTXq/j5adX+fBBQP58Di4dqatZV+zdoqjVOc+PYUJw1fvpxAds52ZN'
                                            'DEen8CqT7Ug0HDBz5FICHXO/Hr1zupXcwmU7eYTdbEFfcJxO4bOwsY'
                                            '8+2KCSTq6CkrmFKDDhLIwKnu4jTVnb6eYdSEGUa296ihHh41MuGSWc'
                                            'dfMnNGLxud6WWTnVtNCmy3q5b+TaiVGz+bte5nM0eOstGyUTaZGuWu'
                                            'wHbfHqN/k46Jx7QABfbMbCCQ0q/NChzq3EJvOreI/X2zAocsxDq+0C'
                                            'Y9TR5qb/LgOks6gTz5MkCBo4/eJpAJ3+Lv+dbMOa+2UO6rLbK22azR'
                                            'zWbeHmmjHZE2+ahunK6vG8dR087Q8mlnJKE0QIGdYxoJ5OfvzFr7nZ'
                                            'lrlrQQSHaLWS0tZg5610Ygz7/hpcDLw7NpRXi27PxfgALPGH+H3hx/'
                                            'R/YfNytwY2wL1ce2SL3NrMAJ79kIpPznoQqc//anBLJyuo9GTffh6L'
                                            'dzac3bufLUDtLOO4i3T6yjHRPrZHRQoLoGBXLM63dp4+t3JWS6WYGr'
                                            'xjRTxZhmGZ+J3kx8/2ktdCatRZb+YtbIX8zszDYazjZ5/1ezrv7VzK'
                                            'vX2mjNWpuEReLuFTmMm0K2EEinuz4KnL8gnwoX5Mv0AlLgWvsbdMX+'
                                            'hnjMCVTg2AN3CWT2bLMCV01ppuopzTI7Gz9nm9mY2UKU2SLf1Jsx98'
                                            'yc/IqNkl6xyT18fhM+3+VDG4G0NwxX4Pae2+hRz23y8iqDuq4ycOzr'
                                            'hegrlKa/SYHDLt6g2RdviPt7gQqcc/8ugfzfArPaLTBz2LJmApkkZg'
                                            '0SMxvyWsgvr0Wybpk18xae36s2qn7VJm/+Zlbg8nU2Kl1nk68nu2jC'
                                            'ZBdO2/slndj7pYT8ZVDgGpdiuuJSLHdeZm18mTkh7iaBjIsNVM/YQA'
                                            '53v0dh7vfk1aX4vKVmjtvcTFs3N8vz580KnHahhUDKG8wKXPOaja68'
                                            'ZhPF5+fg82estxHI4OSRClz18Guqfvi1lHzgq8Uf+HL+/RICeedt1u'
                                            'VvM7d5/0Igk3YHatDuQI5//x4lvH9PxkWZ1TPKzBEHmglE8swKnHy1'
                                            'hfZfbZGq22YFPjzNRiDOrWYFrvrYRhUf2+SDxlG6unEUOw9IJJAHD3'
                                            'zV9sCXQ3eV0exdZZIUxwpsq/uFmut+kQ1HAxU47vQ9Ailbg741ZrbL'
                                            'bCaQCYWYf4VmdrzVQg63WqS0Ee9vI97XN2w0+Q2bxOLzgT0/sRHIB4'
                                            '6uutrRlZf3TqKo3knSK9xP7cP9OMH1AoHsPM0KXLeunm6uq5e6s4EK'
                                            'HN56j+a23hP7j7EefIz1pLKZllU2S1qxWYEdmvH5zS1y5o5ZgXPetB'
                                            'HIL/j8enx+XIyNQB4YR6vNOJojbxwgkJhyrNXlfhyeXkEL0ivknTp8'
                                            '/3XMrr1uEcjyokAFDhrVRBNHNUnXTWYFTr/TTCfuNMujUrMCJz9soZ'
                                            'SHLbLnLub/XTN3n20jEK/f8bx+x/f/qY3KP7VJ2qEBCpyYMomSUiZJ'
                                            '3pujNP/NUWwJ2k0StFtS741T4LYPzhDIyixfjcry5YhrpQTS/VWTAu'
                                            'cfuEq5B67K9dWsdauZwxJ+IZAR5QHqUh7ATzY3UvvmRvn7QKC2H8D8'
                                            '2XOPdu65J3eHmBW4bcN9Aml536y2980ccLyZzMcxn49gPh8xc2ZkC2'
                                            'VEtsioArMCt9W1UGtdi7xba9bltWbuPMxGIHPwvMPwvDNetxFIHtbL'
                                            'fKyXkVgvl2G9nI3vHzgR5/DdOIcfSH1Bk1Nf4PDqaRRRPU3mPRql4Y'
                                            '9GscPMROo9M1F+DPLU9CBPtis4S50Kzkp8m68CP0kro3/TyoTeMSmw'
                                            '69BachtaKy9sYx24jdnx7i8E2G8CsN8EcEZmI53JbJT3UgM1KjWQ7c'
                                            '/eo95n78lQF7wPLmaO2XmfNuy8Ly5rzQocmtVMs7OaxTHVrMCGqBYC'
                                            'WYr5HYn5nYT5vRfzexz2M0/sZ00jbARyBf3A6zHfN2C+22O9B45dYa'
                                            'MtK2wSh37gIMz/8Zj//Y4OVMejAzm9YSaB3P3qZQVOnbmXQIxHPRXY'
                                            'pXsWgXi4+Slw3LpyAonbYVLg8EO1tOBQrZQnswJXeeGe51UvxnsBCm'
                                            'xobiSQ0+mBmpEeyMWX71Hh5XsSMNqswNu/u08gb6wz64x1ZnYvaCa3'
                                            'gmZ57phZux/D/69tIZADRWZNLsL6fbeFnO+2yEP0A6djP8/Afj4N71'
                                            'sI3jfjTBsFzMTzRj9w4kobgWSiH7gN/Y/Q739gkBoPDOL6S29T46W3'
                                            'Zc1IVwW2TE4ikHlPeWn4U16c8No5Apm32E/DF/ux4ysXaMArF+S3dJ'
                                            'O2pps42ulnWuf0s/z+E2vrT1ivN9XT7k318mpbgAa1BXDMgDsEkpsZ'
                                            'qMA5f9wjyx/3xOhhVmDnzPsEsgTrSSTWk4yqZjpVhff5B7zPP6BvQw'
                                            'uBVGB9AU7A+rIT60v6NbMCrx9to3WjMd/Rn49+G9YbkIPoT0Z/apSN'
                                            'jkTZpBH9wElYf0BGfzVYXb8azEFH5hOIi9VVgWM899MGz/3iMsdLgV'
                                            'dvyqZVm7LF6Vs/Ba7pWkEga6tNGl1t4pi9P1Ps3p8lvpQVOLqynlZX'
                                            '1su/doEKnDTpDu2fhPNITqACP7Fvosf2TfK8D95vHzyv0vsUUHpfls'
                                            'WYFbj1ejM9vN4sNTi/AKdtxn62uUUeleD7KMF55tcWAll83azAru44'
                                            'X7hjfcN654n1LikUfaE2aUI/cDrOOyD2WP+AA7D+EdY/h49fVOCQ2U'
                                            'sIpM+U0eowZTS790kmkGMZXgrcdtpCIB9X++n6aj+uLUB/QYWM+sOk'
                                            'wAm9rhHIiZuswOv736J1/W/J4ucCFTB/71D82jsSVIz9sziQo0c34X'
                                            'k1yWV/9Pl39N4nkLexnodhPW+620zNd5vlux/NevhHM0dvb6E121uk'
                                            'XxnWgzIzO/3RQoP/aJH+dfi5DvuLh41yPWxyGP3ARqz3IKMfmNX1Af'
                                            'pXoX+VTfzQD1yP/pvo77HQSYGDWpdTcOtyCfl5tALb3TtIIC1Pe6vt'
                                            'aW/2vGElkNiu/grsGVNJhphK6d6HFDhn8zUCmfSQNeghc9r8W5Q+/5'
                                            'Zs6I/9un8gRx6+Q4sP35HzlYGaUxnIwZOaaPKkJlkdgPNgAN73B/fJ'
                                            '48F98dliVsMWvK+/N9Pe35vlnZNY309iP/+6hUDGlpvVvRzr4V84X/'
                                            '3VIsXoBw72wnncC8/3Hp7vPZyv3sb54m2brER/FPrT3rcRyEL0R6B/'
                                            '/Ua8Lxtt8sorL+n4V17i1MhVlB65Sv6a76bt893YOf0QDU8/JF3meG'
                                            'vnOd4c8T+hxf8TMfj6K7CDdxU9710lQzxIgR3/ukYgS7sEaGSXAHZK'
                                            'vUVDUm/J5GGBCuxUdYcGV92RZ68GavergRwV0UQrIpqkfbxZgTv/e5'
                                            '9AnvrcrJ0/x379pJlAhp3G/nAa6983LbTjmxZ55QLONxfwfP/bQiDv'
                                            '38D3dwP7uY+NQNahHzhsjo1C59jkJPrT0Z+82kYpq23yBfq3o99+k4'
                                            '16b7JJzaAhClybt5ZAnO65KXDtiu/o2orvxO6EtwJnDsghkKPL/DV1'
                                            'mT8n3KuinfeqZGUIaVQIcdqi63Rs0XXp6Rig9o4BXGu7Rddst6TMPV'
                                            'DL3QM5//EdKnx8R0beDFSXm4HcuKGJGjY0yZxJmO+TzLy8RzOt6NEs'
                                            '2dtwv9mG59sVz7dri1zNwP0nw8x1B3B/OdAi0yqwvldg/f9fC4H0u4'
                                            'n5fxO/72cjPz+cr9Bfj/7auTYCaUU/cPwHNgLJQD9wKvqPoP/2wyHa'
                                            '+HAIGzp9TCANkWMUOGToUQKxPcH8f+LN48ecJ5Are/0VOCfhIuUmXJ'
                                            'TH75ICx128TiCmUdjvRgVw/ZgGAulsDFTgCOe7BNLvTqA63gnkjMQm'
                                            'Ajk6xaypU8wcNRj9g5tl4Q7Mzx3os0effYusP2NW4G5HWghkRiX2x0'
                                            'ozt3eyEcg59AMnGrG/GW0yvgnzownn+3AbTQ+3iRvuO24d9601Nope'
                                            'Y5Pr6K9DvyHWRiDxuUMV2LH3pwQy/d4YBbZcSKVzF1Ll8iQfrZnkww'
                                            'ZTLvmZcqWsxF/LS/zZYqqmc6ZqKdlKWryVON+vjgr96qTdP0CBx7/X'
                                            'QOb3GuTPiYHaNhHrffBdArnWEqjAjSfx/E82Se3reL6vYz66NRPIf+'
                                            'PN+iQe95X+LXSlP54v7pOOuE92x33yGdwn36hCfxX2x67Y77vaJBT3'
                                            'S+AmwvmHbHIE/cD583Henm+Td9C/HP1huG+C2D3Efe0h1kv056L/n+'
                                            '3O+mS7Mw/ctYlAtixwV+AZK47R9BXHpDjBR4Et4/NIxufJrr/8NeEv'
                                            'f05uqKb9DdVy8wgpcNTBOgIxTwlQYIeTDWR/skGSXw9U4MTVd2n36r'
                                            'vy/sNAXf0wkFeXNlFUaZM8mInz7kysX6ZmAtn4tVmBg4a2EEhqFuZH'
                                            'Fva7k9jvTrbIaxfNGnwRz/cZ3N+ewfxGP7BLoI2cA23S9T7uA/dxnl'
                                            'iI/XChTVLRD2yMxn4QbZPh6AcO3mwjkMnThylw2DNbCCThmrsCB3VJ'
                                            'o8ld0uTKDR8Fbp+QTyCezkYFjt10iTZuuiSBuaQBucRJXW7Q/i435M'
                                            'W5Aeo0N4CDf2sgkKfDArVbWCB77rtLHvvuSkN7oAIvb2iiZQ1N0vyW'
                                            'WYFdg5tpVDCe/y48/104D7u20EbXFhmD+7w77vNJZ3HePdsipmqcl6'
                                            'qxfz1no7rnbPIZ7vdxuN/PwP0+FPf7CPQDx0bYaGOETW6hvx799rjv'
                                            '98Z9Pwj9wFXor0b/7t7DFdh+VRz1WBUnO6aNVeCqnccJpOswgwJ3Cy'
                                            'ygroEFcmaaUYHXO9UQyJ91pG11xKFLbtCbS27IzhUBCtzZ9TaBjFkc'
                                            'qO6LA7k+5y6BjPpfoAKX/91EpX83Sdkc3F/nYD2a3UxHZjfLgkSzAq'
                                            '/3aqENXi1y3IL7pAX7o7WF3KwtMvIS7geXcJ/tZSOQF26ZdeAt/P4E'
                                            'nPcn2ETQD1y7COvfIpv0x33fEff9RvSDLEM/cMgWzJ8teB8qhitwVP'
                                            'lWAtl+fqwCD+z3I4EsW2ZQwH5XSCBO64wKnJZVQyD/PCJ98gjrf+EN'
                                            'OlZ4Q77ZEKCJGwLYuOQ2+S25LQ5RgQp8+M5dSrlzV17tgu+/C9Zj+/'
                                            'uUYH9fvp5v1oT5Zi6ObKbSSKz/e7H+7zXzeGohECcxK3Bybgvtz20R'
                                            '+xrs7zXYzx1sdMLBJkfRn4r+hFdttPNV3Gea0duM+b/ERkOW2CQE/c'
                                            'AZ67BerMP+h/7t6K9BfzX6U2NHKHB6/y/oVP8v5KC7hya7e3D4znRa'
                                            'sDNdup0wKHDAsCICWXvIqNGHjOw07TKBzLNnDbfHeWfoTQLx34H1f0'
                                            'cA2x28Tf8m35bsDwPV8mEgr3/6Hq17GvedZ3HfeRafP/w+VQ+/LxMX'
                                            'mRU4cU0zgTz1Lfb/bzE/J7ZQxcQW2ab4exXn+aIWAvkd/a3of9LPRo'
                                            '/74bzbgPNuA76fSdjfJ9nkXfQvR39kpI0WR9pkC/qBYz7C/PjIJsfR'
                                            'n9bx/D/Deekzm6x9x0uj3/Fi95+yyeOnbPmrm0Hbuxm488186nozX2'
                                            'zD/BTYbk45dZpTLmuv+mv0VX9uW1FNIBHPjlPguk9OE8jwHzwVOLN/'
                                            'Fp3pnyUPX/RW4O6jhHqMEpm4wkeBg625BLL1JW8FTh4jBHJ6pY9mrP'
                                            'ThkNxcAonaZ1Dg6NuFtO52ocxN9VXg1G9L6ci3pZKZZFBgu6ZCAtn0'
                                            'va/Gfu/LOQdLSQ7i//f6KbDx3wsEYngf57f3/Tm860UCOb/PT3P2+X'
                                            'Hr/1UQyPDV/gqc/vRFOvX0RVnubVRgY8El8iu4JLcajVrfaOQ6/ysE'
                                            'Unt6rAKHN52guU0n5Og+D03d58FJdqcIZNjaceq8dhwHPcygiQ8z5K'
                                            '+Jnto+0ZPt9SyB5D3rpfnPenFG5Dk6FXlOBhV66cBCL7a0WkhaLZKw'
                                            'xluB80OVCkNVTg3yUeDxKedpYsp5GRA9ToET/syg3X9mSMOrngpszD'
                                            '1LIAe6e2lydy9Ofwfjv3NOHmB8G8ZP/MNCIG+v9dawtd68/m0lkBWD'
                                            'fRS47vB5Akmz+GD98WGXi3k0/GKebJ9uUOCcWwUE8vFzXrr+OYy/Au'
                                            'OvOCdS5KXAUY8stOKRRYZEeyuw51wl97kqri/6KHDU0fO06uh5mWP1'
                                            '0TCrDydeyiOQcW8Y1PMNA9s3FlDvxgIZfMugwK7XisjtWpFsW+Kr25'
                                            'f4ssvZEhp1tkTuYfwmjN8YrgRyD+M3YXy7Y+ep07HzYi8+Ctz9Sh6B'
                                            'XMf4dRg/4m4BgQjGBw6vKyKQoZG+6hzpy+OzSsicVSLv3fPVqHu+HL'
                                            'ChjEC6h/spcKzrBQLJxfjAsbV5tLE2TzbPMGjsDANnNhUQyJwGg4Y1'
                                            'GDjxZhGBZGH8TIwfml1Cs7NLpFOTrwIHxGD8mDJJwPjAy90u0DK3C1'
                                            'KB+yZwW2EFgXRjfwVunVxFIE8wPnBVfRGV1xfJ60t9NWSpL6+2ltAa'
                                            'a4nEY/z4jvE3lhFtLJOB8/0UONj9Ak11vyBTL/kpsKW4gqS4QjZifO'
                                            'DDwVV0JLhKuh7xV2A760X613JRRv3HqMDFAy4RyOD7vgpsjC0jkBSM'
                                            'D+zucYE8PC5IFcYH3l5aQTtKK+QJxgcOnVpFs6dWyTqMD5wsFwlkN8'
                                            'YHDhl4iUCSFhoVODSyhmZH1kj1OaMCJxy6TImHLsu0Gj8NqfHj0PIK'
                                            'AvkgwB/3SdzPQqrIPqRKbBgf2JhzkQJyLkrnbkYFrh10ia4NuiSdI/'
                                            'BzhJHLl9ZQxdIa8czG+SbbyE8OX6bHhy9L9XMmBfasukIgL4WZ1CnM'
                                            'xEG2qwQy9h93df/HnWvWpdGVdWmStHasAju/cIKGvHBCMn8dq8DGkz'
                                            '8SyIEF2F8WeHCY6SSFmk7KokseGnHJg9P1FJ1SrBeBWC8Cx3GcIYO2'
                                            'GjLkwQ/j1PbDOO6e+hOBXO7jqTV9PHFfOksgf0R7alu0J9veyaRf38'
                                            'mUWdc9NfQ61t+cLLLkZEk/fy919Pfi7s9l0zPPZcv4RC8FdnjdQr1f'
                                            't8ivD70U+Ml2K/273SqjXvNWYLvzQiBdDuL+exD3nwdKQQ9UdrVhPW'
                                            'rz5m69zxMI/zZWA34by91P/0jdTv8ozgs9FHggnySQe+hrQl/U+VME'
                                            'koI+wH0tg7z9MmRx2jgFTv/+Jzrx/U8yxcFTgx08uZv9Wepqf1aWfO'
                                            'ipkR96cvS7mbTu3UwpQV8x+sbnZhHIJvTFoi+uRzaBnEAfcMx0C22c'
                                            'bpE3//BS4IgdVgLZgT7g8FyhsFyRpeiLRF/yr0ogf6KvDX2hfc4TiH'
                                            'mijwKnuefSCfdcWbjLRyN2+XD9xDy6OTFPIms8FNgl7xSBDDaPU+BG'
                                            '/wwCuYW+evSF/PATTf/hJ8lDXz76knudpZReZ+U6+urQ131lJoH41n'
                                            'mqoc6T8/OyqDAP+yf6gDvbZxOI4zd4nt94cfsbFnr0hkWs6LOgr+pL'
                                            'K5V/aZWH6AO25AlJnkgp+oDbf1MCmfGntwIfdjhPRxzOywH0JaOvfW'
                                            'wuPR6bKxb0AQ8MyiMQSwN+bvBh46x8oln5cmy0QYHj5hUQyNvHx2nY'
                                            '8XHclvYTPUr7Sfz6eiqwsc9Z8utzVqasw/Nc58mJUZkEkoo+4OCCLJ'
                                            'pckCVzjV4KvL5XNm3olS0b0QccMdNCi2daZFiblzq3ebHhKyt5f2WV'
                                            'WcHeGhrszQMLhAZj+XZN8Vbg4N+Vpv6uchp9Gehr63ueHvY9L/+g7w'
                                            'n6AsblEki/3T7quNuHQyfl0exJedLnto863Pbh1aH5tCo0X+zcDAqc'
                                            'GY71O7xAWtYY1LbGwMELCgnkltWg9VbsF3OKaO6cIjmPvhz0uazKpF'
                                            'GrMqX/DU91vOHJVYVZVF2YJeXoA27vnU2Pe2eLDX3AdW9a6NqbFtmO'
                                            'PuDknVYCyUafBX3rC4U2FIpsRx9w8kPMz4cqfR55q8MjvI/9zpO533'
                                            'l561UfDX3Vh2M8c2mjZ66sQh9wwuQ82j05T95B33L0pb2VTyBvog+4'
                                            'aX4BNc4vENNagxrXGjhqYSGtWlgo4wT7rxg4Zm4RbZhbJIFP+WrAU7'
                                            '7s+kYxub1RLNETfRU4LbCEjgWWiJ/JS4FXO2TTGodseXMv3re9Xhwc'
                                            'aiGQNvQBd0uw0jMJVnGa4q3A2GoJpBV9wO1/KD3+Q+U99EWhL97xPI'
                                            'H8hL4M9BV75RJIOfqAa1/Loyuv5Uke+vLRVz87n+pm58sP6EtDn8NC'
                                            'nB8WFshX6ItHX2JEIYFsQl8s+hLnFdHeeUXyFfri0Rc0o5gmziiWfP'
                                            'QBZ5pL6Jy5REbH+aprnC87v1xKw18ulctFvlpThPPnc2V07LkyeftP'
                                            'Lw37E+edXVZatcsqm9EXi76AEiEqEZl+yFuBQ/5UArmIvir01fXH+a'
                                            'r/eekd5KPA9j65BDJ8j48C20/Jo95T8mRgo48C24flk0MY5ucYzM8x'
                                            'uI9EFBBFFEgj+oAtiwoJ5BL6aqRj/hZh/hbJL+irR1/YzGIC6f2qrw'
                                            'KXv1JCINHoA3Z3LSUQp2JfBU7vUUYgvbr6qX1XP45oKaPIljIJN/sp'
                                            'cFxuOW3NLZdT6AM+/EgJxP0vbwV2feE8gUShDzjUkEsgMegDDp6aRy'
                                            'Cr0AccNCefQN5EH3DkogJavqhAfKINaojGfF1cSI2LC2WYGtRZDVw7'
                                            'v4hAxnT2VffOOO+8WUxr3iyWeegLR1/d+BICOY++HPQZRpcSyBL0RX'
                                            'b09URfzzJ5C32h6Iu0ldFiW5kcRh9wTF45bcwrl+kf+ymw064LBLLq'
                                            'jJ8CL59fQZHzK6QMfeXoS/TNpb2+uXINfcDbp+URSBn6ytEXMzcf71'
                                            'O+pKEPOGlxAe1dXCBb0QfcPbKQukUWyhr0AbcvKKJHC4rkY/StR1/c'
                                            'rGICOYY+YNuEEmqeUCLdP/dV4AC3UjK7lcqP6EtHX4Z9GYEkoy+54/'
                                            'k9KKPwB2XShD7g1fnlBLIbfcD2uy9Q790X5DT6MtAXuqCCZiyokMm/'
                                            '+ilwxMhKWjyyUhY7+ytgPlTSgpZKGXrHR53v+HD6vHw6NS9f/s8d89'
                                            'PdwMVLCqh0SYHcQF8d+tyXFpLb0kIpRh+wQ0QRPR9RJGXoK0ff4dBi'
                                            'Sgktlnb0Adu9WkJPJpbILPSFoi9oTClNHlMq7egDzuxVRmd6lckd9D'
                                            'WiL+zXMpr7a5m4veKnwMsLymlZQbn8jL5a9D1Bn92eC/IH+trQZ1xY'
                                            'QSCfoy8OfUGjKgnkMPqAjbZKCrBVytS3/BV4/PdVBLJzu78Cr59/ka'
                                            'LnX5SxHxrU/UMDz1hWSCD9cgzqmGNgz0VFBNK/i686dsHzeKuYMt8q'
                                            'lglBvjo+yJftg0qoe1CJHERfMvpC3UtphnupBJZgvS3B7/dGX2/0/A'
                                            'c9/0Hfb2UE8j76VqMvvLCcFhSWywsbcJ7f4Me2PRcIxOusn3qe9WOX'
                                            'iAoCKUYfsMvLlTTq5Uq5hb569Dk9qKSBDyplG/q2o2/gsSoafKxKyt'
                                            'EHHLTgIoGki78C5/StJpCA3/xx3sN5Pb+aQJaiLxJ9VbOLCSQefcAD'
                                            'J5UQyAP02dAXNraUQD5HXxz6MvuU0bk+ZbIafcAzWssIJBN9wKFF5f'
                                            'RmUbnMQ184+uoTLxDIGvQB2y+qIJBuv/kpsL1rJfVwrZQXh/mr0zB/'
                                            'foI+kAL05aOvDX0gXb/AfeYL9C68SAMWXpT76GtC3/bnq2nH89USjT'
                                            '7gyIJqAtHBRs0ZjPvByksEkhtsVOCqvjVU07dGfLb6qmGrL0d6lNJi'
                                            'j1KpRh+wxaGMQM6h71xH3+9lFPJ7mTxBH3BwcTlNLi6Xg+hLRl/5Nx'
                                            'cI5Cz6MtHXjr7H6AtCH3Ab+h6ibw76wtBX+2slXfu1Up6a7a+dZ/tz'
                                            '+Q9VVPFDlQSgD9iGvl/RN0T9FTisXzXN7lctP6IvHX3uhdXkVlgtbe'
                                            'gDjo+6RDuiLsnf6GtHX/zzNbTz+Ro5/yH6P8T9K7OGZmTWyNvfGTXs'
                                            'OyO3zrxMbTMvS6dufgoc/LCMpj4sk4DxfgrY/8rJXFIut9HXiL7MvR'
                                            'fo3N4L8jf62tHXuLiCGhZXyGfoi0NfzehKqh5dKfvQl4S+zN8q6cxv'
                                            'lWJCnxF9h9OqKCWtSj5C33r05URcJJAw9AG7OFbTcMdqaUIfcDv6Hq'
                                            'Fv1Iu4r75o5BnvXaLp712Sl6cY1XUK7pf9amhqvxppQx+wfVYN9ciq'
                                            'kS/Qtx19qW9eJpB/Ko36pNLIhtbLBDL1b6MC126+QiAun/gpcNq+Cw'
                                            'Til+mnwOVLKgikCH3F6Mt0qySQWvQBJ7RWEshH6FuPvvXHqwjkDPqA'
                                            'ExZdpJ2LLsou9CWgrw19IC+2Yn63or+omqSoWuahLxx99qsuUe9Vly'
                                            'QcfcDdHdHjWCMj1hnVZR3u6+grRJ+gDzh01mWaPeuyjK7C91Fl5Cb0'
                                            'gXyMvvXoi91yhUAyXjIpcIjjVZrqeFWuB5m0LsjEM45cJZAurX7aud'
                                            'WPk8ZUEki/4f7qONyfo36vpFW/V8oZ9AHPOFFF009UyW/oa0VfxOKL'
                                            'BFKOPuCc/lhf+lfLm+gDji2upi3F1bIHfYnoq0UfSAL6gIvRV46+t9'
                                            'EXhr6YczUE8jv6WtHnEHqZeodelrnoA078/TKBpKEP2P2zK+T22RW5'
                                            'jb5G9LWh7yH6uk0yKXA7+kCGrTSp80r0jq2l0LG1wgkmDUgwsUtmLY'
                                            'H8P6Z6tC8='
                                        ),
                                    },
                                ],
                                [
                                    'theirs',
                                    [
                                        {'float': '0x1.13d35c0000000p-2'},
                                        {'float': '0x1.000a060000000p-1'},
                                        {'float': '0x1.000c120000000p-1'},
                                    ],
                                ],
                            ],
                        },
                    },
                },
            },
            'x86_64': {
                'research': '7440581b1',
                'recorded': '2026-10-23',
                'jax': '0.6.1',
                'jaxlib': '0.6.1',
                'numpy': '1.26.4',
                'xla_flags': ['', '--xla_cpu_multi_thread_eigen=false'],
                'values': {
                    '21ef8819b7ed7ae6e5ae970485f80e9b28fcdb8ab3533a8d186f0a1fbb534dd4': {
                        'tests': [
                            (
                                'test_hmr_names.py::TestSingleNetworkEvaluators::'
                                'test_equal_to_hmrs_on_hmrs_arrays'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 10,
                                        'substrate': {
                                            'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'max_depth': 3,
                                            'variance_threshold': 0.03,
                                            'neuromodulation': {
                                                'mode': 'true_neuromodulation_4nt_option_a_tanh',
                                                'modulation_strength': 5.0,
                                            },
                                        },
                                        'neat_species': {
                                            'compatibility_threshold': 2.5,
                                            'max_stagnation': 40,
                                        },
                                    },
                                },
                            },
                            {
                                'problem': '_TwoInputXOR',
                                'get_data': [
                                    [[0.0, 0.0], [0.0]],
                                    [[0.0, 1.0], [1.0]],
                                    [[1.0, 0.0], [1.0]],
                                    [[1.0, 1.0], [0.0]],
                                ],
                                'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                                'input_shape': [2],
                                'inputs': [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
                                'jitable': True,
                                'output_coords': [[0.0, 1.0]],
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                            },
                            42,
                            [0, 4, 9],
                            '<f4[4]',
                            '<f4[4, 2]',
                            '<f4[4, 1]',
                        ],
                        'value': {
                            'dict': [
                                [
                                    'W1',
                                    {
                                        'array': '<f4',
                                        'shape': [3, 2, 340],
                                        'zlib64': (
                                            'eNrt1M8rgwEcx3F/wXMkxUFLy0FL1A6y55OkOTxWflwMscOs0FJKHE'
                                            'TDgWgXWZliK482Bxaz+fHsu112WHKR/Dg5KMtl0g7kwN/wXPd5Xd7/'
                                            'wdu5kkLQ5YYtvIremmZJlg2M71zg0h5BsV5DbCQNq0fHVsiH/HyrY+'
                                            'wqjmjnEsq7g6q1qUN6guv46btW3xvqZGYyJFVERBWq/dbA6Ns+AoUY'
                                            'dMWm3rmzuPcb8FUn4Xrcg2LJYLs2hXgiCvvrALRCGs6DI8x6/BjWzh'
                                            '3P+WOEWwLwWZbVl3Abf0pEFcs2F5HNk4Q8KbrEv8/4QyIik6aLOVko'
                                            '5cT78d/PHH9KRGTSg/dQGqVbvrpOpTC1xp8SEZn0258VY/FGJkpZGd'
                                            'rI8KdERCb9AZZic0U='
                                        ),
                                    },
                                ],
                                [
                                    'W2',
                                    {
                                        'array': '<f4',
                                        'shape': [3, 340, 1],
                                        'zlib64': (
                                            'eNrbaTPhwMRfqQfWh9k5HL/V5PDXbNUBrQ2LD/gXzT7gKzbxQE188g'
                                            'EHBpsDV9/yHsi6unm/zVQm+/x7G+0LuXgd1ltZO9S9bHa4/WCCw3L3'
                                            '2Q4hfYsdGEbBKBgFo2CEAkWD/QdunDt04F3C/gNPLhw6MBoio2AUjI'
                                            'JRQB7I2bHtwEnWAwdsNLYdWPp4/2h5OgpGwSgYBWQCALz+PII='
                                        ),
                                    },
                                ],
                                [
                                    'bg',
                                    {
                                        'array': '<f4',
                                        'shape': [3, 340],
                                        'zlib64': (
                                            'eNoV1nlY1VUawHFTTERFJRcCzCVFJULcFdGvICBuiIQIyHpBdu6+Af'
                                            'feg0pmWmbmPo4ZKpFjZo4L5TKmlGuIZNogmvtouTBomubYvP1xn/s8'
                                            '7+/5nN857znve34FY7P5dX5vJvvp2V05ke11+ayY8hY7H7xJWWgbPM'
                                            'J0HDaksv3ueAjswkc6E6+tmkfj/mn8q7EfkX+3MuxZIZVRc+jbeQTT'
                                            'hhdx6O0sduycQyv/aOp3j+BpfS8897kR/E03Es5oOTEwl5fXpbB+YQ'
                                            'yOJaGsu+iPK9eTrYM7cMPbgDWtgOwXGbwUG0/LuUmcdQ7h4r2ezDa8'
                                            'xAFvExnDiwm+k83FZUk0LZvK8arRXB/SF6a5433BTKudOtZo8ugelc'
                                            'aRvbF4T4XYHgHkPOjK4rlWtJcMDDYXEuPUUB6TwOH0KDa+P4xF6b60'
                                            'bLcxodaEa6CWSeNzqEyey63r01m7KISUBf1ZfdaOrsjC/bf1vPVBPi'
                                            'evppNyPg774zDuh72JJqaYirfzOeI5j/hz6Sz5JomltXE46qfy6Ocw'
                                            '0p+Mxm/gm1S+1xd9XHfidrjzsK8bfS51YfOkXhhXaWl2L+Tc+Ryi72'
                                            'TyZ0AKDYtnc+DhDIZmRfL3C+OoTBnKNg9/NO18WFHRCdvBVuR36kjE'
                                            '37zZcUyHzVzE+YV5eF3JYqEpjSeDE1ndEku3I9EMXgV/GEfyTBtA/W'
                                            'evcTvWizkb2nJhRDtmXX2F0bf0RJ0o5nmPAi59NY8eGzKoqU4m6/u3'
                                            'yH8+jcAhk1iTN5bwfUGcHvU6zZ174mnzoDmhNX94d6Zfi4F33XXYPy'
                                            'zkvC2Xuq0amtumUl2RQIjXTLpsjSIwdDzzrw5j6z8GUnjIl85TO7Nr'
                                            'cWvWR3rgumske7ge9aiIIN98TnyYzYrwdDq+mkRtmzjKH0+h5u5Env'
                                            '46ihttA5mS1IfrXbthyWlHVUFbYs6baJlqIDRMi9/WAoZl5rAtPZPk'
                                            'pXMpOBuP5vUZuEoiCP4xhIRJwXx5qT9DT3lzaEJHvlzaivd2momONd'
                                            'Jo19HQo4iQ7nlMT8vC2pRKbdkczrwRy4urk3n+twm8ohlBmymDOW3s'
                                            'hbtnV7a/5cYghwVthInSDXoWGIrp9kU+m5hHSqcMFrZJpluXt3g2YB'
                                            'o6wrFkjOHW2jfZ17YfN+q78+OQ9riFWMkZZGbk5wYS92p5NKSQBrdc'
                                            '/HprcOSnsK1+NgWRMdw8FEmf8aGEnRpK0Hx/OlT4MKqtJ89+seL/Pz'
                                            'Pe243Yb+nwXFeE1xd5nPPKZuOONIZoE3FMnMXvr06h1RMIvzySbdcC'
                                            'SO/fm1n1XjR+YGPtUQtB6020aWPgg1+LWUMB1Xfmce+HDObdSWZZz3'
                                            'hU7HQi3p9Eev1Ybg8cwpMdr/Pt+z0pGGxng8OKzm6mUycjWwbpWLmv'
                                            'kNNbcnE2agiYlMqgUwlUaGaS/mcUiZvG80vMcAb0HURiiB/pNXbaD7'
                                            'ZRP9nC5rYmtLP0jPEuRhuZj/5oNjZHOg/nJjElNo4X06cydEYYz2aP'
                                            'xmEP5M+GPhSNL6H6hI1R7a0cv2eifa6BPRotx04X0PxBDvGrMzl8ci'
                                            '7jfGbT1TqDYecjqB43joZdwfwnfgBjviohJN1O+UEr5u/MLC4wErxW'
                                            'R/tQOY9heTR+mIXJO437h+ewuzSWWyHRnH4xgYl1Iyg9OJhlBjsLzi'
                                            'mef1tCUpNiR2IRjhsOxryrKLqv2PiKlidBTowrFT2aFeHbdCw1Otmw'
                                            'UbH2v0r22cBPXzhp+FTxqEUR6JnJiwEmnt4qxc/NxblxCvNuRfM1xQ'
                                            '+PFC8yNbQcNNFtQBkr/FzERcoYNYrkG4q03xRen2Uxeqac6Ywymoa7'
                                            'SIqR9+xX7LypuCbPv76ejVejmZq1Zbw81UVLguLOIcWTW4qMx4ppXj'
                                            'nUplkoryujf7qLPumKrkdkXrcVF+S598hcllyyUPuSg/EmF7/kyDu/'
                                            'VSTcUUx5oiiZnsfWBCsbhzlIXyTz0yruHlNof1HskedHEvOJOWklIN'
                                            'PBmrUuMiyKLScVpb/KOn5X7H03iU/25DBtj47OVRY0ZSUk/lpGqtmJ'
                                            'qcDF4zaKhVmK65K/I0cVlY2KczK3SsllTxn//pxkwiNzSeyuZ/kNC4'
                                            'mbS1jv5mBkhZNks4t+Hoo5uYrITxShMu9lFxU5Mvfn4mPEL+83l3YN'
                                            'ucSZ9FT1tZJ2soSJvRxs/8hJa5cL/y6yj4WS082Kz79TGGTPH4mPl7'
                                            '17R/yWO3N5MTePiDN6klOtlDaXMGukg9pKJ9p3XLj1UOyXnFzcquh+'
                                            'XBF3SeGU3GwTf0D80M9TCL6ex9jBBnatsbKreyl10x24vnSyabmLg7'
                                            '6KqUbJebWcixOyL5cVL0vuXoh/ID68WHpvbj4h8w3srbfiEVLKwSwH'
                                            'l/7lZN06F/F9FbWS85BtilOS91Y/K5aIn/lQ0U/yf3lwGvo7skcXDG'
                                            'jdbbybWkrvUgc3v5f8Vbo45q/wL5H1ble8elpxRrzHXcXH4uPFd7me'
                                            'xqq8AsrfMHJ5go1h5aU8/cDByotObm5z4R2osDjkrO1QpH+vWH9F8i'
                                            'b+nvgK8VfWptPnZgFNTiNdzTY6VJYyY6uD5ttOQv7pInqo1JhSaHZK'
                                            'bdXJGFdlDfcUY6U2dolPnJaBb3ohOXVGOlTb6HG0lNe+dvD7b05S9r'
                                            'tIGCXnZYGc111ST2cUfaSuSsQvEH9VfPmzDFacLySgl4mGRhsp10vR'
                                            '1jnY1drFtKPyfqnHG28rJkhNtj6raBJ/T/wp8Z2eKrKrMlk2rYjgAh'
                                            'P6jnYuty4j4poD384u3E/J+idKHS6WOtqrCGhQrLgua5A+4SW1OVa8'
                                            '50wNfoeKmL/HxO1QO5V9y/j4NwejfF2sbnBxOkJyvlRxe58i+gdFhN'
                                            'R9nfgE8dniFz3SEBFUzIA/TYQV2dlOGRXuTtwHunj0b6m3KYrqZZLD'
                                            'rxSZ0rceig95oFgn/n3xP63OwmtDMf6TzVSst/MipYxHPk6WD3PR/6'
                                            'qLxhkK/YeKXOkpph8VG6SvbBbfJH73X/Mflc3ydlqWvG9m1zE7m0rK'
                                            'aAl0cna8i0G3XYyOU/T/qxceVLjOy/yl73hIX/SV3tIkfs1oAz2vmf'
                                            'jZ28qmOzZ2N2kZY9Oz+p6BleEmRjww0FHu2zdK5b5ZKX2qzEzEKgv9'
                                            'Vlnl+85G6Gr5d9moibWT0bmE88uKWavTkhelY/Mreqov6Dm7woAt0s'
                                            'iDB0aSJ+t4r5ueRT/p2fKRgbAoIzubjUR/ZGJukJmsf+uxrTTgMdmI'
                                            '8b9GWq004THEzOUjZk7MtPCDxIe2GKmReI3EK46ayYu1sOO8hU3SB/'
                                            'WrTOQFmwmsNfPyLAtRFywEz7HS8ayVXpE2GiReJfHLEj8s8VUS3yTx'
                                            'vrttJPWyY/nJwqxEK30arLweZWOjxM9IvGW+nfCrdqolXiVx3z02ol'
                                            '6zk7XAzg6Jp4SWULWihAk/F3EivpivjhXjM0buoq1aMrrqSC3V8eyK'
                                            'johIPZM/1dOxnYHKbANdDhvQvCr3uN7IlW+NhB4vZr04a5WWRnE/it'
                                            'Nd1bFXXI24UnEd5xmwizsuzt1gJPw7I+WvmjgjNfBI3GwvHTPKdNwU'
                                            'FxClZ1C1ntvi7OKuiAvyMVIkbou4K+ICC018+LUJH3cz34pLvqZjg7'
                                            'g14ma7G7gmLvobOQ/i6sW5HTMy0cfEYnF3xZnEDYw347XRTOfJetw/'
                                            '03NUXEyO9FRxz8QNMxrJEbdR3C1x0ftNnBbnFJcnbsUtM+5vyl61N1'
                                            'AjzueIgQxfI6vFHRf3P3GTikx8Iq5/ezPnxO0Xd1Ncorg+RgsReyws'
                                            'EHdM3DNxg44bSfU18bG45+Ic4gJmm+nysZlR/zFTLU4rboU4nycWuo'
                                            '22kmoyskTcAXHPxSUcMHFO3Hxx2eKWiHsq7oC4y+JyxKWIO2Wx8vUu'
                                            'Kz5+JpKKTXwpLsjDzAVxNeKaxE0JstDBZOGNvRZ2itsizstqpdU/rd'
                                            'juWykfZKOjfGssEheUYMZzk/zfNrNeXLI4u7g/xN0XlywuStx+cUfE'
                                            'aTJtbF1ro0KcRly5uF/EVYurFRf1u4UxY+Ssi3tP3AtxbvIduEXcDX'
                                            'Hf19nIa2PnB3EhQyw8Ftdln4Xl4srF3RF3Vlz4A3mvuBvi+q2T++KM'
                                            'jCHu8Sg7Hvl2oswW0sQ1iTspLthmpcduK++Ie1dcf43MVVy0uCviho'
                                            '22M0mcn/S+oyfs+D+14DXWilVciriT4k6Jyxa3SdwicX5udorEOcTF'
                                            'i3smzvq7nboBJRwSVynOs1l+ATaqxF0RVysuUVyVuL3ilooLPmnnH+'
                                            'Ja+0vdxpWgcZYQJy5W3C1xvdfbeKnexkpx18TdF3dUXJ64K+JGidOL'
                                            'Wynu009L+LK+hP8DCiuvEA=='
                                        ),
                                    },
                                ],
                                [
                                    'rd',
                                    {
                                        'array': '<f4',
                                        'shape': [3, 340, 4],
                                        'zlib64': (
                                            'eNoV131czff/x/F+hq9tRsjEsEaIJUkXpzrn9aozLBaaMc1CCDGjmd'
                                            'HMrBFrmDULjUZiNIuFRumc1ytdX6kkNGJJSB1bs2bN1/b9Pfvj/kd0'
                                            'e58e5/P+vC9i4kbwJ3Ej1HH4FzRg+BfS9PxaAvH8ZIMYPtlAadM8+f'
                                            'g0T3W9cZZAxn/zK4HYnU7xh+zVltH8vmW0Jm45SHu3HJTea3szaIhz'
                                            'fwp27i8GSieQmh09pXZHT4rpU2kB4/CpDeIytYFOvDeOQaN/yaB1v2'
                                            'RI/MYX+auNL2r4+iUUsX6JZHS9SyBO1bEWu+pYY+0pO6o5ZSeT689I'
                                            'cP0Z6rzXh5/a66Ous/LIZVaeTO8zkkEjR+ykZSN2yt0//8OgMT7ZJr'
                                            'AG9I4jkAifRRLps4hCU/wYNLZ/BW3sXyFN3d0ZNCP4ezoT/L0EWPox'
                                            'aIybgUCi1xbQmrUFEh8jVjB1nuTOoJ0dj1Enx2MScm44g3b/YiuBJE'
                                            'xw5K8nOGrrAn9qW+AvKyydGNRu3EZTu8dGa9yxItpyrEhC/DOsYCqe'
                                            'uoxA7L7e3IECqp43gXXj5hsSu/kG1Sc6C1Ddj0cEyPKPB2f/46FtIa'
                                            'cJ5NTlUQyaeH8P7b2/R6ZGD2bQnO7zydJ9vtg3PcugTsZ7JrBG7b5F'
                                            'IAFv3LCAcf2l3QRiv2eidNszkZxi7pgcY+5YIzpbBShn+CkrmIwDiw'
                                            'WoIdiLQesnZ9PNydkyeJYbg5ZvO0Sl2w6JS9FQHlk0VNOcPiWQ2Ct9'
                                            'ePOVPhoy/CUKHv6SRH/cTiA5I/80gsUp4ycanPGTBMx6qgNV+Zs6SD'
                                            'rtkzTaRzmpBktMqsFoeaZJgGIn+DCovVsu9XbLlfoZYxnUdvg4NR8+'
                                            'LoUhLgxafOJLKj/xpXgeG8BexwaoU+kkci6dJEmvdOV9r3TV1m2pJr'
                                            'DGx18kEPuFm61gevLzpwQS//N7HSgmNtYE1t56URz0IkW3GXhtm0Fz'
                                            'BheTDC6W4IZx/FoD5vPCM7Rm4RlJd3flk+6umrYgiY4tSJKQzU78+m'
                                            'YnrS94l0DiH/XgLx/10Nq2zgTSFNZMIHZNP2WDf3RcCq2JSxH3pNHi'
                                            'kjSaUrf3oLTtPcTvVLoYT6VTaoQfH43w0yfGC/TYeEGe9PTm//b01h'
                                            'kPrQR4h8ewzydjdPuKowQyKm4Ygyaf2UIgjdHPM6jjD+7k8IO7ODna'
                                            '8UuOdmp34TVTa/lr1vqQHAIJDbtiDQ+7YmrNmEUgp1J2SHrKDgo/4M'
                                            '/zDvirw3cXqfd3F2WAzYdBA6LzCSTudQ/+7HUPre90km52OilBVSMZ'
                                            'NCg+gUBcHwxk0PgvZ1HCl7OkNelpBk3efsEEVrvu1wkkudcYK5jixu'
                                            '0gENd9oeKyL5Ri0owMWrXyMlWsvCyFG30Z1NavlFr7lUpzlieD1gdn'
                                            '0c3gLOkcPZpBg1OSCaTHB0MYNLPpQwLJv96LQYv396X8/X2lm+Uhgc'
                                            'R4f2QEy/La4wSSUO8g8fUO1BY1lv+MGqsxnU/Qhs4nJPLd0QzauiGZ'
                                            'bBuSJfIjFwb19I0nkPaIIfx3xBCN2vQhgcQ2D+BNzQM0Z9prlD/tNU'
                                            'nP68Un83ppQJ++BJKa0pWPpnTV9P1ppsz9aVb3NQ/JYw3+nj9XGu0e'
                                            'rbQkJ1VTSlK1tM6NsYIp+sPjBJJwqq8knupLdZ6bOkjMwJUdqFvnkR'
                                            '3k1NHDknH0MKWf2GoCa/3qKmlcXUXJHp2tYApZ+osAxQ/pKkC1H2QI'
                                            'kHHwYjEMXkxBBz+T8Qc/o3W5HgyatvkUgUx+bgyDOv72HYFYPhrF1o'
                                            '9GqX3ubgKJuO7MoHU7YglkcusgBo0qnEOrC+eI5wd92euDvurywygC'
                                            'KR70LJcMelZjZt80Rc2+ae1m+IdAnIJdTWCNnvELgUTZdbHG2HUxRX'
                                            'pZCKR27H1r/dj7ptorXxNI6o3gDhQQOYWMkVMkLQH/lvA1ua+/agLr'
                                            '3N45Et47h+zWWv0hu9OEZuk8oZncJ+RYwWR+vkDGP19AEWvMErnGTB'
                                            'VFiVJVlEjjH49jUJczZwhkzRfuDJr/v2NU+L9j4jDTlfvMdFXD9STy'
                                            'vp4kLkNG8IghIzT1+jY6fH2bRFqcGDRx0EpKGrRSHt9yZFDH7kwgVU'
                                            'E9GTSorCuFlHWVpJSnGDTk/jYTWDtPbiEQu+Mzs+1OzPTv5lZKz7iV'
                                            'SuqKVGv6ilTTwKJDBNLaPKoDuWRGdZCpSzdK8NKNFPJ5rw4yefaPEj'
                                            'z7Rwp/d5wp6t1x1tJvrkn5N9coJmyxFUwJsy8LUFDhIAFKrfhegEIG'
                                            'ePG0AV7que8ceew7JzNax/IbrWM148yPBBL6jBu/9Yyb1m9NoYatKZ'
                                            'JpHMlZxpEacWInLTixU8qnD+Wy6UM1PPwTCgv/RKb2GMigOabplG+a'
                                            'LnYufRg0edQgSho1SGKmd2PQeB81gTXk00cEYpdTZnTKKbM0zb9Kd+'
                                            'dflRjrdCuYop6cIpCQqKclNOppco/8nDwiP5fVHy2QqI8WUPsW7w7i'
                                            'sHB/B0ruddQEVrfoUnGNLqX6s7ssOWd3GQ+73ZFUtzsUt6rVGrOq1f'
                                            'T423Py5NtzFOXszSudvdXiKwQyxDCOQbf3z6Ad/TNk/P4x/Mr+Mep5'
                                            '9Sh5Xz0qYW4v89tuL2vkc99QxHPfSMDZYRx4dpiGTfmMZk/5TBb8O5'
                                            'hBwydFEEimPs9n9XkNmeZBIKk/duejP3bX1v+0muy6tVpbP7Hj3z6x'
                                            '04D3Zpjc35thrb14m0DqH6Rbch6kG8ujzhNIvOtFK5i6lX5DXUu/kd'
                                            'A3AjpQ++ezO0jw+C86UFX0Hyawpv6RJUABI1KMTiNSLAu+bJSILxsp'
                                            '/n+HrGBqG1wmQKEDfHjWAB9dv/s8Re8+L22LPRk0bXQmnRidKeF9xv'
                                            'K8PmPVtTCN3ArTxOGxK/d97Kp2/9tP//67X8IejOA5D0bo+uU7CGTi'
                                            'mZcYNMr3A1ru+4GMdxjAExwGaCpPJJDxYs+viL1GufYkkFGLuzBoVd'
                                            '8DJrCWx/xGIHaZNn+7s7Zs+5cqCcTQ6Sure6evTGFrvieQxs8GS/1n'
                                            'g6l12ccdxME1Whxdoyku8MUOkvrS9x3Iadtyk/u25dY1Z2ok+kwNhf'
                                            'iNtob7jTZ5G6+Lp/E6xf/jw1/946O1P+XTlZ/yJXODF2dt8FL39mzy'
                                            'aM+Wrss9GDR2zkkCyU9xY9CIzw/Tgs8PS0z7SP6kfaTW99lFIHW+zg'
                                            'xq2b6RQFpHDuLfRg7S+qfeoqan3pLkjQ58cKODhvYbRjP6DROnv55m'
                                            '0NYPL5rAGvH+YwKpd+ppqnLqaU3oXkcgdq8Ms4LJsCiTvBdlSvFnj6'
                                            '1Vnz02LY/9kkCK62d2oNp7r3SQAf0TZWD/RIp6I88EVvsV+QJkd3Jb'
                                            'NviHNDTLjIZm2l1pYNDuSUXUI6lIDJ96s++n3vrkY6V/P1aJODaOFx'
                                            '0bp/kpPxGI/cvu3PNld0099T2B1Fx/mS9ff1nrJ+2jhkn75Nr24Qxa'
                                            '/PhzKnz8udzs7MSg9d8to9rvlonT/zkyaGaJL2WU+Eq9dw/+xbuHhv'
                                            '9oRyDFszpx0axOGvDrRyb3Xz+ydvdqIpAYOw8LGJO+KiSQxLAsa3JY'
                                            'lin+8AECOXx5nCRfHkehsUs6SMXvW6Tq9y1kd+g/1PnQfySq9HQHso'
                                            't8xmQf+YzVvfimuBbfpFNf+TJo5yml1GlKqThE+XCfKB9NzsullLxc'
                                            '2VLlyYD1MYsM+7Jk+a6x/O4urI8xJygz5oSE7h3Ns/aO1mKHg1TocF'
                                            'ByTrnw+VMu6vggnp5/EC/23w/hnt8P0Uj3jwikG73AT9MLGjd5CoEc'
                                            'GdmbQeMW9KPtC/qJ6+D/MGiM7aQp2nbSar//DwKxq/3SWH/1S0vk35'
                                            'do8d+XJPno+1YwNT37I4EE+PSWIJ/e1NYe2wHvxjsdKPq6awc5fCtF'
                                            'km+lUJVlpwmsnfIrpHN+BTlP8OOhE/w0sryclpWXS3mwgS8EGzT6Qg'
                                            'Gtu4A9s96LQZtes9Ld16zi0ujBIxs91PPvUwRS4T2GQS3rcJdZd0Tq'
                                            'T4xiULtle6jzsj0SOmAY1pdhmjprMx2btVmSTIMZND1lHoGUX+rLZZ'
                                            'f6aubPrpTxs6vkxD/L5+Of1eRFt01gtQT/SyB2Vh+TvdXHWpVUTyAB'
                                            'R1otYEw+baX9p3FOnttgDZjbYLLF7SKQ2qmTOlC68+uU5vy6tJV+1Y'
                                            'GqVvxiAuvOWSIJs4S6tfgxaOagSjo3qFKqHX0ZNPVAMYGEXvfmWde9'
                                            'dXtWDm3NypGI/p68sL+nJr50lkAqTrszaM6nPxBISKwrvx7rqrVrv6'
                                            'Vra7/FuXsEzwoboVF7thNI61Mv8W9PvaTRdVEEcti3P3/n21+TBwRS'
                                            '6oBAST3dk78/3VOjxzxNq8c8LTK0M4NWhX9lqg3/yhqywUZTN9jErv'
                                            '3LLDu7eN/8PWUEElVw0Aqmbou/I5CMxSMkffEISp65uoNE7Y/pQFEr'
                                            '+9LqlX3lyp7jUrsHZ+FYf3431l/dx18kkIjLvrzosq+2mcrooalMwv'
                                            'J8+O08H81oyqNTTXli7+PFPX28tPv/zhGI9PFg0NQx6XRkTLo88XLj'
                                            'f7zctHHgIWoYeEjWrxrJH68aqeVzviaQtoSh/EfCUI3/NIZ2fBojG+'
                                            'YNZFCX0TMIxGNbHwYtfvdFAqkq7cYXS7tp6/J8U/3yfGv3lL8IJMB8'
                                            '22hnvm1xPFtLA87WSk75BCuY1n+WQSDtj7uI3X+7UNWwbVQxbJtENM'
                                            '2VyKa51D7Dr4Ms2Jck4fuSaHxvI0/obdQa8yUCydzkx6BPfrhAdmkX'
                                            'pPAbA4Mm3Cik3TcKxY69GTR2t9DG3SK1oeMYtPvWDHpma4YkFI5hUM'
                                            '95qQTSNfxlBh148hsacPIbiWwbxkvbhmlwrzgC2TLpRQZNHbAYz3+x'
                                            'ZLzQj0+/0E/tVnnSk/c8JeHF5xg05nGbKe5xm7Wu6//x9a7/p/Wn5p'
                                            'lyTs2z5ve+QyDhd/ZYYu7sMcbcyyWQ1CWl1vQlpSaXD/YRSGORUeqL'
                                            'jBTlP7eDXPtwm9R+uI1W7zIy6OqBlwmkydmf7zljPjyoJLcHlfJoni'
                                            '+DGpJKyDupRBo8fBg01XaeDtvOi9tmTwYN2J1JIEmGsQyaM/445Y4/'
                                            'Lu0jRjPo4T0HKGXPAWkY5sKgyUO/pJSh2Lf++xIX/fclrXqwhmoerJ'
                                            'GYpQP4k6UDNNEcRCDG4b3YNLyXRsX2IpBOjV0Y1P6Dw6ZuHxy21hxv'
                                            'JRC7PcONYDFsqiLvTVUSsmyrNWDZVlP3Hj8QyOFuL0hqtxfIYUNMB8'
                                            'nZvboDlfcwcVkPk2ZeukJnLl2R1ix//i3LXw93rqbkztXi0s+PR/bz'
                                            '01CXcgIxOxgYtK5LAYEc2evFoBFvWWjBWxZZvsODQYMunSSQ4RfcGD'
                                            'Sp33cEkuk6is+6jtIm2UXNskuc1jnzS+uc1eGtTfT8W5vkWvQgBnVq'
                                            'mk0gTrcdGDRm6giKnTpCQpY/w9OWP6Ppu6+YwOrwxX8JJPz0ABNYg2'
                                            'bfIJDWaEer3YeOpqqmLAKxP/aHFUxh/l8RyJNpOJuHTKdun5gY1P6d'
                                            'WgLJfcXIoDP+vEQgcTl+/FmOn8ZsriCQ7Y0G/qLRoDPuFlHo3SLpus'
                                            '+bQcfXKgXVqriWjuPRpeM02P4MTbY/I97T3RnUyecYDfTBfao77lPd'
                                            'XTXq/j5adX+fBBQP58Di4dqatZV+zdoqjVOc+PYUJw1fvpxAds52ZN'
                                            'DEen8CqT7Ug0HDBz5FICHXO/Hr1zupXcwmU7eYTdbEFfcJxO4bOwsY'
                                            '8+2KCSTq6CkrmFKDDhLIwKnu4jTVnb6eYdSEGUa296ihHh41MuGSWc'
                                            'dfMnNGLxud6WWTnVtNCmy3q5b+TaiVGz+bte5nM0eOstGyUTaZGuWu'
                                            'wHbfHqN/k46Jx7QABfbMbCCQ0q/NChzq3EJvOreI/X2zAocsxDq+0C'
                                            'Y9TR5qb/LgOks6gTz5MkCBo4/eJpAJ3+Lv+dbMOa+2UO6rLbK22azR'
                                            'zWbeHmmjHZE2+ahunK6vG8dR087Q8mlnJKE0QIGdYxoJ5OfvzFr7nZ'
                                            'lrlrQQSHaLWS0tZg5610Ygz7/hpcDLw7NpRXi27PxfgALPGH+H3hx/'
                                            'R/YfNytwY2wL1ce2SL3NrMAJ79kIpPznoQqc//anBLJyuo9GTffh6L'
                                            'dzac3bufLUDtLOO4i3T6yjHRPrZHRQoLoGBXLM63dp4+t3JWS6WYGr'
                                            'xjRTxZhmGZ+J3kx8/2ktdCatRZb+YtbIX8zszDYazjZ5/1ezrv7VzK'
                                            'vX2mjNWpuEReLuFTmMm0K2EEinuz4KnL8gnwoX5Mv0AlLgWvsbdMX+'
                                            'hnjMCVTg2AN3CWT2bLMCV01ppuopzTI7Gz9nm9mY2UKU2SLf1Jsx98'
                                            'yc/IqNkl6xyT18fhM+3+VDG4G0NwxX4Pae2+hRz23y8iqDuq4ycOzr'
                                            'hegrlKa/SYHDLt6g2RdviPt7gQqcc/8ugfzfArPaLTBz2LJmApkkZg'
                                            '0SMxvyWsgvr0Wybpk18xae36s2qn7VJm/+Zlbg8nU2Kl1nk68nu2jC'
                                            'ZBdO2/slndj7pYT8ZVDgGpdiuuJSLHdeZm18mTkh7iaBjIsNVM/YQA'
                                            '53v0dh7vfk1aX4vKVmjtvcTFs3N8vz580KnHahhUDKG8wKXPOaja68'
                                            'ZhPF5+fg82estxHI4OSRClz18Guqfvi1lHzgq8Uf+HL+/RICeedt1u'
                                            'VvM7d5/0Igk3YHatDuQI5//x4lvH9PxkWZ1TPKzBEHmglE8swKnHy1'
                                            'hfZfbZGq22YFPjzNRiDOrWYFrvrYRhUf2+SDxlG6unEUOw9IJJAHD3'
                                            'zV9sCXQ3eV0exdZZIUxwpsq/uFmut+kQ1HAxU47vQ9Ailbg741ZrbL'
                                            'bCaQCYWYf4VmdrzVQg63WqS0Ee9vI97XN2w0+Q2bxOLzgT0/sRHIB4'
                                            '6uutrRlZf3TqKo3knSK9xP7cP9OMH1AoHsPM0KXLeunm6uq5e6s4EK'
                                            'HN56j+a23hP7j7EefIz1pLKZllU2S1qxWYEdmvH5zS1y5o5ZgXPetB'
                                            'HIL/j8enx+XIyNQB4YR6vNOJojbxwgkJhyrNXlfhyeXkEL0ivknTp8'
                                            '/3XMrr1uEcjyokAFDhrVRBNHNUnXTWYFTr/TTCfuNMujUrMCJz9soZ'
                                            'SHLbLnLub/XTN3n20jEK/f8bx+x/f/qY3KP7VJ2qEBCpyYMomSUiZJ'
                                            '3pujNP/NUWwJ2k0StFtS741T4LYPzhDIyixfjcry5YhrpQTS/VWTAu'
                                            'cfuEq5B67K9dWsdauZwxJ+IZAR5QHqUh7ATzY3UvvmRvn7QKC2H8D8'
                                            '2XOPdu65J3eHmBW4bcN9Aml536y2980ccLyZzMcxn49gPh8xc2ZkC2'
                                            'VEtsioArMCt9W1UGtdi7xba9bltWbuPMxGIHPwvMPwvDNetxFIHtbL'
                                            'fKyXkVgvl2G9nI3vHzgR5/DdOIcfSH1Bk1Nf4PDqaRRRPU3mPRql4Y'
                                            '9GscPMROo9M1F+DPLU9CBPtis4S50Kzkp8m68CP0kro3/TyoTeMSmw'
                                            '69BachtaKy9sYx24jdnx7i8E2G8CsN8EcEZmI53JbJT3UgM1KjWQ7c'
                                            '/eo95n78lQF7wPLmaO2XmfNuy8Ly5rzQocmtVMs7OaxTHVrMCGqBYC'
                                            'WYr5HYn5nYT5vRfzexz2M0/sZ00jbARyBf3A6zHfN2C+22O9B45dYa'
                                            'MtK2wSh37gIMz/8Zj//Y4OVMejAzm9YSaB3P3qZQVOnbmXQIxHPRXY'
                                            'pXsWgXi4+Slw3LpyAonbYVLg8EO1tOBQrZQnswJXeeGe51UvxnsBCm'
                                            'xobiSQ0+mBmpEeyMWX71Hh5XsSMNqswNu/u08gb6wz64x1ZnYvaCa3'
                                            'gmZ57phZux/D/69tIZADRWZNLsL6fbeFnO+2yEP0A6djP8/Afj4N71'
                                            'sI3jfjTBsFzMTzRj9w4kobgWSiH7gN/Y/Q739gkBoPDOL6S29T46W3'
                                            'Zc1IVwW2TE4ikHlPeWn4U16c8No5Apm32E/DF/ux4ysXaMArF+S3dJ'
                                            'O2pps42ulnWuf0s/z+E2vrT1ivN9XT7k318mpbgAa1BXDMgDsEkpsZ'
                                            'qMA5f9wjyx/3xOhhVmDnzPsEsgTrSSTWk4yqZjpVhff5B7zPP6BvQw'
                                            'uBVGB9AU7A+rIT60v6NbMCrx9to3WjMd/Rn49+G9YbkIPoT0Z/apSN'
                                            'jkTZpBH9wElYf0BGfzVYXb8azEFH5hOIi9VVgWM899MGz/3iMsdLgV'
                                            'dvyqZVm7LF6Vs/Ba7pWkEga6tNGl1t4pi9P1Ps3p8lvpQVOLqynlZX'
                                            '1su/doEKnDTpDu2fhPNITqACP7Fvosf2TfK8D95vHzyv0vsUUHpfls'
                                            'WYFbj1ejM9vN4sNTi/AKdtxn62uUUeleD7KMF55tcWAll83azAru44'
                                            'X7hjfcN654n1LikUfaE2aUI/cDrOOyD2WP+AA7D+EdY/h49fVOCQ2U'
                                            'sIpM+U0eowZTS790kmkGMZXgrcdtpCIB9X++n6aj+uLUB/QYWM+sOk'
                                            'wAm9rhHIiZuswOv736J1/W/J4ucCFTB/71D82jsSVIz9sziQo0c34X'
                                            'k1yWV/9Pl39N4nkLexnodhPW+620zNd5vlux/NevhHM0dvb6E121uk'
                                            'XxnWgzIzO/3RQoP/aJH+dfi5DvuLh41yPWxyGP3ARqz3IKMfmNX1Af'
                                            'pXoX+VTfzQD1yP/pvo77HQSYGDWpdTcOtyCfl5tALb3TtIIC1Pe6vt'
                                            'aW/2vGElkNiu/grsGVNJhphK6d6HFDhn8zUCmfSQNeghc9r8W5Q+/5'
                                            'Zs6I/9un8gRx6+Q4sP35HzlYGaUxnIwZOaaPKkJlkdgPNgAN73B/fJ'
                                            '48F98dliVsMWvK+/N9Pe35vlnZNY309iP/+6hUDGlpvVvRzr4V84X/'
                                            '3VIsXoBw72wnncC8/3Hp7vPZyv3sb54m2brER/FPrT3rcRyEL0R6B/'
                                            '/Ua8Lxtt8sorL+n4V17i1MhVlB65Sv6a76bt893YOf0QDU8/JF3meG'
                                            'vnOd4c8T+hxf8TMfj6K7CDdxU9710lQzxIgR3/ukYgS7sEaGSXAHZK'
                                            'vUVDUm/J5GGBCuxUdYcGV92RZ68GavergRwV0UQrIpqkfbxZgTv/e5'
                                            '9AnvrcrJ0/x379pJlAhp3G/nAa6983LbTjmxZ55QLONxfwfP/bQiDv'
                                            '38D3dwP7uY+NQNahHzhsjo1C59jkJPrT0Z+82kYpq23yBfq3o99+k4'
                                            '16b7JJzaAhClybt5ZAnO65KXDtiu/o2orvxO6EtwJnDsghkKPL/DV1'
                                            'mT8n3KuinfeqZGUIaVQIcdqi63Rs0XXp6Rig9o4BXGu7Rddst6TMPV'
                                            'DL3QM5//EdKnx8R0beDFSXm4HcuKGJGjY0yZxJmO+TzLy8RzOt6NEs'
                                            '2dtwv9mG59sVz7dri1zNwP0nw8x1B3B/OdAi0yqwvldg/f9fC4H0u4'
                                            'n5fxO/72cjPz+cr9Bfj/7auTYCaUU/cPwHNgLJQD9wKvqPoP/2wyHa'
                                            '+HAIGzp9TCANkWMUOGToUQKxPcH8f+LN48ecJ5Are/0VOCfhIuUmXJ'
                                            'TH75ICx128TiCmUdjvRgVw/ZgGAulsDFTgCOe7BNLvTqA63gnkjMQm'
                                            'Ajk6xaypU8wcNRj9g5tl4Q7Mzx3os0effYusP2NW4G5HWghkRiX2x0'
                                            'ozt3eyEcg59AMnGrG/GW0yvgnzownn+3AbTQ+3iRvuO24d9601Nope'
                                            'Y5Pr6K9DvyHWRiDxuUMV2LH3pwQy/d4YBbZcSKVzF1Ll8iQfrZnkww'
                                            'ZTLvmZcqWsxF/LS/zZYqqmc6ZqKdlKWryVON+vjgr96qTdP0CBx7/X'
                                            'QOb3GuTPiYHaNhHrffBdArnWEqjAjSfx/E82Se3reL6vYz66NRPIf+'
                                            'PN+iQe95X+LXSlP54v7pOOuE92x33yGdwn36hCfxX2x67Y77vaJBT3'
                                            'S+AmwvmHbHIE/cD583Henm+Td9C/HP1huG+C2D3Efe0h1kv056L/n+'
                                            '3O+mS7Mw/ctYlAtixwV+AZK47R9BXHpDjBR4Et4/NIxufJrr/8NeEv'
                                            'f05uqKb9DdVy8wgpcNTBOgIxTwlQYIeTDWR/skGSXw9U4MTVd2n36r'
                                            'vy/sNAXf0wkFeXNlFUaZM8mInz7kysX6ZmAtn4tVmBg4a2EEhqFuZH'
                                            'Fva7k9jvTrbIaxfNGnwRz/cZ3N+ewfxGP7BLoI2cA23S9T7uA/dxnl'
                                            'iI/XChTVLRD2yMxn4QbZPh6AcO3mwjkMnThylw2DNbCCThmrsCB3VJ'
                                            'o8ld0uTKDR8Fbp+QTyCezkYFjt10iTZuuiSBuaQBucRJXW7Q/i435M'
                                            'W5Aeo0N4CDf2sgkKfDArVbWCB77rtLHvvuSkN7oAIvb2iiZQ1N0vyW'
                                            'WYFdg5tpVDCe/y48/104D7u20EbXFhmD+7w77vNJZ3HePdsipmqcl6'
                                            'qxfz1no7rnbPIZ7vdxuN/PwP0+FPf7CPQDx0bYaGOETW6hvx799rjv'
                                            '98Z9Pwj9wFXor0b/7t7DFdh+VRz1WBUnO6aNVeCqnccJpOswgwJ3Cy'
                                            'ygroEFcmaaUYHXO9UQyJ91pG11xKFLbtCbS27IzhUBCtzZ9TaBjFkc'
                                            'qO6LA7k+5y6BjPpfoAKX/91EpX83Sdkc3F/nYD2a3UxHZjfLgkSzAq'
                                            '/3aqENXi1y3IL7pAX7o7WF3KwtMvIS7geXcJ/tZSOQF26ZdeAt/P4E'
                                            'nPcn2ETQD1y7COvfIpv0x33fEff9RvSDLEM/cMgWzJ8teB8qhitwVP'
                                            'lWAtl+fqwCD+z3I4EsW2ZQwH5XSCBO64wKnJZVQyD/PCJ98gjrf+EN'
                                            'OlZ4Q77ZEKCJGwLYuOQ2+S25LQ5RgQp8+M5dSrlzV17tgu+/C9Zj+/'
                                            'uUYH9fvp5v1oT5Zi6ObKbSSKz/e7H+7zXzeGohECcxK3Bybgvtz20R'
                                            '+xrs7zXYzx1sdMLBJkfRn4r+hFdttPNV3Gea0duM+b/ERkOW2CQE/c'
                                            'AZ67BerMP+h/7t6K9BfzX6U2NHKHB6/y/oVP8v5KC7hya7e3D4znRa'
                                            'sDNdup0wKHDAsCICWXvIqNGHjOw07TKBzLNnDbfHeWfoTQLx34H1f0'
                                            'cA2x28Tf8m35bsDwPV8mEgr3/6Hq17GvedZ3HfeRafP/w+VQ+/LxMX'
                                            'mRU4cU0zgTz1Lfb/bzE/J7ZQxcQW2ab4exXn+aIWAvkd/a3of9LPRo'
                                            '/74bzbgPNuA76fSdjfJ9nkXfQvR39kpI0WR9pkC/qBYz7C/PjIJsfR'
                                            'n9bx/D/Deekzm6x9x0uj3/Fi95+yyeOnbPmrm0Hbuxm488186nozX2'
                                            'zD/BTYbk45dZpTLmuv+mv0VX9uW1FNIBHPjlPguk9OE8jwHzwVOLN/'
                                            'Fp3pnyUPX/RW4O6jhHqMEpm4wkeBg625BLL1JW8FTh4jBHJ6pY9mrP'
                                            'ThkNxcAonaZ1Dg6NuFtO52ocxN9VXg1G9L6ci3pZKZZFBgu6ZCAtn0'
                                            'va/Gfu/LOQdLSQ7i//f6KbDx3wsEYngf57f3/Tm860UCOb/PT3P2+X'
                                            'Hr/1UQyPDV/gqc/vRFOvX0RVnubVRgY8El8iu4JLcajVrfaOQ6/ysE'
                                            'Unt6rAKHN52guU0n5Og+D03d58FJdqcIZNjaceq8dhwHPcygiQ8z5K'
                                            '+Jnto+0ZPt9SyB5D3rpfnPenFG5Dk6FXlOBhV66cBCL7a0WkhaLZKw'
                                            'xluB80OVCkNVTg3yUeDxKedpYsp5GRA9ToET/syg3X9mSMOrngpszD'
                                            '1LIAe6e2lydy9Ofwfjv3NOHmB8G8ZP/MNCIG+v9dawtd68/m0lkBWD'
                                            'fRS47vB5Akmz+GD98WGXi3k0/GKebJ9uUOCcWwUE8vFzXrr+OYy/Au'
                                            'OvOCdS5KXAUY8stOKRRYZEeyuw51wl97kqri/6KHDU0fO06uh5mWP1'
                                            '0TCrDydeyiOQcW8Y1PMNA9s3FlDvxgIZfMugwK7XisjtWpFsW+Kr25'
                                            'f4ssvZEhp1tkTuYfwmjN8YrgRyD+M3YXy7Y+ep07HzYi8+Ctz9Sh6B'
                                            'XMf4dRg/4m4BgQjGBw6vKyKQoZG+6hzpy+OzSsicVSLv3fPVqHu+HL'
                                            'ChjEC6h/spcKzrBQLJxfjAsbV5tLE2TzbPMGjsDANnNhUQyJwGg4Y1'
                                            'GDjxZhGBZGH8TIwfml1Cs7NLpFOTrwIHxGD8mDJJwPjAy90u0DK3C1'
                                            'KB+yZwW2EFgXRjfwVunVxFIE8wPnBVfRGV1xfJ60t9NWSpL6+2ltAa'
                                            'a4nEY/z4jvE3lhFtLJOB8/0UONj9Ak11vyBTL/kpsKW4gqS4QjZifO'
                                            'DDwVV0JLhKuh7xV2A760X613JRRv3HqMDFAy4RyOD7vgpsjC0jkBSM'
                                            'D+zucYE8PC5IFcYH3l5aQTtKK+QJxgcOnVpFs6dWyTqMD5wsFwlkN8'
                                            'YHDhl4iUCSFhoVODSyhmZH1kj1OaMCJxy6TImHLsu0Gj8NqfHj0PIK'
                                            'AvkgwB/3SdzPQqrIPqRKbBgf2JhzkQJyLkrnbkYFrh10ia4NuiSdI/'
                                            'BzhJHLl9ZQxdIa8czG+SbbyE8OX6bHhy9L9XMmBfasukIgL4WZ1CnM'
                                            'xEG2qwQy9h93df/HnWvWpdGVdWmStHasAju/cIKGvHBCMn8dq8DGkz'
                                            '8SyIEF2F8WeHCY6SSFmk7KokseGnHJg9P1FJ1SrBeBWC8Cx3GcIYO2'
                                            'GjLkwQ/j1PbDOO6e+hOBXO7jqTV9PHFfOksgf0R7alu0J9veyaRf38'
                                            'mUWdc9NfQ61t+cLLLkZEk/fy919Pfi7s9l0zPPZcv4RC8FdnjdQr1f'
                                            't8ivD70U+Ml2K/273SqjXvNWYLvzQiBdDuL+exD3nwdKQQ9UdrVhPW'
                                            'rz5m69zxMI/zZWA34by91P/0jdTv8ozgs9FHggnySQe+hrQl/U+VME'
                                            'koI+wH0tg7z9MmRx2jgFTv/+Jzrx/U8yxcFTgx08uZv9Wepqf1aWfO'
                                            'ipkR96cvS7mbTu3UwpQV8x+sbnZhHIJvTFoi+uRzaBnEAfcMx0C22c'
                                            'bpE3//BS4IgdVgLZgT7g8FyhsFyRpeiLRF/yr0ogf6KvDX2hfc4TiH'
                                            'mijwKnuefSCfdcWbjLRyN2+XD9xDy6OTFPIms8FNgl7xSBDDaPU+BG'
                                            '/wwCuYW+evSF/PATTf/hJ8lDXz76knudpZReZ+U6+urQ131lJoH41n'
                                            'mqoc6T8/OyqDAP+yf6gDvbZxOI4zd4nt94cfsbFnr0hkWs6LOgr+pL'
                                            'K5V/aZWH6AO25AlJnkgp+oDbf1MCmfGntwIfdjhPRxzOywH0JaOvfW'
                                            'wuPR6bKxb0AQ8MyiMQSwN+bvBh46x8oln5cmy0QYHj5hUQyNvHx2nY'
                                            '8XHclvYTPUr7Sfz6eiqwsc9Z8utzVqasw/Nc58mJUZkEkoo+4OCCLJ'
                                            'pckCVzjV4KvL5XNm3olS0b0QccMdNCi2daZFiblzq3ebHhKyt5f2WV'
                                            'WcHeGhrszQMLhAZj+XZN8Vbg4N+Vpv6uchp9Gehr63ueHvY9L/+g7w'
                                            'n6AsblEki/3T7quNuHQyfl0exJedLnto863Pbh1aH5tCo0X+zcDAqc'
                                            'GY71O7xAWtYY1LbGwMELCgnkltWg9VbsF3OKaO6cIjmPvhz0uazKpF'
                                            'GrMqX/DU91vOHJVYVZVF2YJeXoA27vnU2Pe2eLDX3AdW9a6NqbFtmO'
                                            'PuDknVYCyUafBX3rC4U2FIpsRx9w8kPMz4cqfR55q8MjvI/9zpO533'
                                            'l561UfDX3Vh2M8c2mjZ66sQh9wwuQ82j05T95B33L0pb2VTyBvog+4'
                                            'aX4BNc4vENNagxrXGjhqYSGtWlgo4wT7rxg4Zm4RbZhbJIFP+WrAU7'
                                            '7s+kYxub1RLNETfRU4LbCEjgWWiJ/JS4FXO2TTGodseXMv3re9Xhwc'
                                            'aiGQNvQBd0uw0jMJVnGa4q3A2GoJpBV9wO1/KD3+Q+U99EWhL97xPI'
                                            'H8hL4M9BV75RJIOfqAa1/Loyuv5Uke+vLRVz87n+pm58sP6EtDn8NC'
                                            'nB8WFshX6ItHX2JEIYFsQl8s+hLnFdHeeUXyFfri0Rc0o5gmziiWfP'
                                            'QBZ5pL6Jy5REbH+aprnC87v1xKw18ulctFvlpThPPnc2V07LkyeftP'
                                            'Lw37E+edXVZatcsqm9EXi76AEiEqEZl+yFuBQ/5UArmIvir01fXH+a'
                                            'r/eekd5KPA9j65BDJ8j48C20/Jo95T8mRgo48C24flk0MY5ucYzM8x'
                                            'uI9EFBBFFEgj+oAtiwoJ5BL6aqRj/hZh/hbJL+irR1/YzGIC6f2qrw'
                                            'KXv1JCINHoA3Z3LSUQp2JfBU7vUUYgvbr6qX1XP45oKaPIljIJN/sp'
                                            'cFxuOW3NLZdT6AM+/EgJxP0vbwV2feE8gUShDzjUkEsgMegDDp6aRy'
                                            'Cr0AccNCefQN5EH3DkogJavqhAfKINaojGfF1cSI2LC2WYGtRZDVw7'
                                            'v4hAxnT2VffOOO+8WUxr3iyWeegLR1/d+BICOY++HPQZRpcSyBL0RX'
                                            'b09URfzzJ5C32h6Iu0ldFiW5kcRh9wTF45bcwrl+kf+ymw064LBLLq'
                                            'jJ8CL59fQZHzK6QMfeXoS/TNpb2+uXINfcDbp+URSBn6ytEXMzcf71'
                                            'O+pKEPOGlxAe1dXCBb0QfcPbKQukUWyhr0AbcvKKJHC4rkY/StR1/c'
                                            'rGICOYY+YNuEEmqeUCLdP/dV4AC3UjK7lcqP6EtHX4Z9GYEkoy+54/'
                                            'k9KKPwB2XShD7g1fnlBLIbfcD2uy9Q790X5DT6MtAXuqCCZiyokMm/'
                                            '+ilwxMhKWjyyUhY7+ytgPlTSgpZKGXrHR53v+HD6vHw6NS9f/s8d89'
                                            'PdwMVLCqh0SYHcQF8d+tyXFpLb0kIpRh+wQ0QRPR9RJGXoK0ff4dBi'
                                            'Sgktlnb0Adu9WkJPJpbILPSFoi9oTClNHlMq7egDzuxVRmd6lckd9D'
                                            'WiL+zXMpr7a5m4veKnwMsLymlZQbn8jL5a9D1Bn92eC/IH+trQZ1xY'
                                            'QSCfoy8OfUGjKgnkMPqAjbZKCrBVytS3/BV4/PdVBLJzu78Cr59/ka'
                                            'LnX5SxHxrU/UMDz1hWSCD9cgzqmGNgz0VFBNK/i686dsHzeKuYMt8q'
                                            'lglBvjo+yJftg0qoe1CJHERfMvpC3UtphnupBJZgvS3B7/dGX2/0/A'
                                            'c9/0Hfb2UE8j76VqMvvLCcFhSWywsbcJ7f4Me2PRcIxOusn3qe9WOX'
                                            'iAoCKUYfsMvLlTTq5Uq5hb569Dk9qKSBDyplG/q2o2/gsSoafKxKyt'
                                            'EHHLTgIoGki78C5/StJpCA3/xx3sN5Pb+aQJaiLxJ9VbOLCSQefcAD'
                                            'J5UQyAP02dAXNraUQD5HXxz6MvuU0bk+ZbIafcAzWssIJBN9wKFF5f'
                                            'RmUbnMQ184+uoTLxDIGvQB2y+qIJBuv/kpsL1rJfVwrZQXh/mr0zB/'
                                            'foI+kAL05aOvDX0gXb/AfeYL9C68SAMWXpT76GtC3/bnq2nH89USjT'
                                            '7gyIJqAtHBRs0ZjPvByksEkhtsVOCqvjVU07dGfLb6qmGrL0d6lNJi'
                                            'j1KpRh+wxaGMQM6h71xH3+9lFPJ7mTxBH3BwcTlNLi6Xg+hLRl/5Nx'
                                            'cI5Cz6MtHXjr7H6AtCH3Ab+h6ibw76wtBX+2slXfu1Up6a7a+dZ/tz'
                                            '+Q9VVPFDlQSgD9iGvl/RN0T9FTisXzXN7lctP6IvHX3uhdXkVlgtbe'
                                            'gDjo+6RDuiLsnf6GtHX/zzNbTz+Ro5/yH6P8T9K7OGZmTWyNvfGTXs'
                                            'OyO3zrxMbTMvS6dufgoc/LCMpj4sk4DxfgrY/8rJXFIut9HXiL7MvR'
                                            'fo3N4L8jf62tHXuLiCGhZXyGfoi0NfzehKqh5dKfvQl4S+zN8q6cxv'
                                            'lWJCnxF9h9OqKCWtSj5C33r05URcJJAw9AG7OFbTcMdqaUIfcDv6Hq'
                                            'Fv1Iu4r75o5BnvXaLp712Sl6cY1XUK7pf9amhqvxppQx+wfVYN9ciq'
                                            'kS/Qtx19qW9eJpB/Ko36pNLIhtbLBDL1b6MC126+QiAun/gpcNq+Cw'
                                            'Til+mnwOVLKgikCH3F6Mt0qySQWvQBJ7RWEshH6FuPvvXHqwjkDPqA'
                                            'ExZdpJ2LLsou9CWgrw19IC+2Yn63or+omqSoWuahLxx99qsuUe9Vly'
                                            'QcfcDdHdHjWCMj1hnVZR3u6+grRJ+gDzh01mWaPeuyjK7C91Fl5Cb0'
                                            'gXyMvvXoi91yhUAyXjIpcIjjVZrqeFWuB5m0LsjEM45cJZAurX7aud'
                                            'WPk8ZUEki/4f7qONyfo36vpFW/V8oZ9AHPOFFF009UyW/oa0VfxOKL'
                                            'BFKOPuCc/lhf+lfLm+gDji2upi3F1bIHfYnoq0UfSAL6gIvRV46+t9'
                                            'EXhr6YczUE8jv6WtHnEHqZeodelrnoA078/TKBpKEP2P2zK+T22RW5'
                                            'jb5G9LWh7yH6uk0yKXA7+kCGrTSp80r0jq2l0LG1wgkmDUgwsUtmLY'
                                            'H8P6Z6tC8='
                                        ),
                                    },
                                ],
                                [
                                    'theirs',
                                    [
                                        {'float': '0x1.13d35c0000000p-2'},
                                        {'float': '0x1.000a060000000p-1'},
                                        {'float': '0x1.000c120000000p-1'},
                                    ],
                                ],
                            ],
                        },
                    },
                },
            },
        },
        'agree': True,
    },
    'hmr_reference.frozen_trajectory': {
        'platforms': {
            'arm64': {
                'research': '7440581b174be13887db6c391e45efc9c3ee2668',
                'recorded': '2026-10-22',
                'jax': '0.6.1',
                'jaxlib': '0.6.1',
                'numpy': '1.26.4',
                'xla_flags': ['', '--xla_cpu_multi_thread_eigen=false'],
                'values': {
                    '66596304bcad6d7cc82a259902592c6027df212e9ee19b369020c37ea48cbcc2': {
                        'tests': [
                            'test_hmr_reference.py::test_emr_reproduces_hmrs_trajectory[sin]',
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 60,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'variance_rule': 'legacy',
                                            'max_depth': 2,
                                            'variance_threshold': 0.03,
                                            'dynamic_functions': {
                                                'mode': 'global',
                                                'hidden_activation': 'sin',
                                            },
                                            'extra_randkey_split': True,
                                        },
                                    },
                                },
                            },
                            {
                                'problem': '_Parity3',
                                'get_data': [
                                    [[0.0, 0.0, 0.0], [0.0]],
                                    [[0.0, 0.0, 1.0], [1.0]],
                                    [[0.0, 1.0, 0.0], [1.0]],
                                    [[0.0, 1.0, 1.0], [0.0]],
                                    [[1.0, 0.0, 0.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [0.0]],
                                    [[1.0, 1.0, 0.0], [0.0]],
                                    [[1.0, 1.0, 1.0], [1.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_coords': [[-1.0, -1.0], [0.0, -1.0], [1.0, -1.0]],
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 0.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 0.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 0.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_coords': [[0.0, 1.0]],
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0], [1.0], [0.0], [0.0], [1.0]],
                            },
                            42,
                            3,
                        ],
                        'value': {
                            'tuple': [
                                [
                                    {'float': '0x1.7fb1540000000p-1'},
                                    {'float': '0x1.8685ec0000000p-1'},
                                    {'float': '0x1.ea37440000000p-1'},
                                ],
                                {
                                    'array': '<u4',
                                    'shape': [2],
                                    'zlib64': 'eNoT3P9NvnyWxGkAEM0D2g==',
                                },
                            ],
                        },
                    },
                    'f7bfa31137fdcef710f2c6c8daf1a3ab69a227b2e71533b277ba9cd7b8e8009c': {
                        'tests': [
                            'test_hmr_reference.py::test_emr_reproduces_hmrs_trajectory[tanh]',
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 60,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'variance_rule': 'legacy',
                                            'max_depth': 2,
                                            'variance_threshold': 0.03,
                                            'dynamic_functions': {
                                                'mode': 'global',
                                                'hidden_activation': 'tanh',
                                            },
                                            'extra_randkey_split': True,
                                        },
                                    },
                                },
                            },
                            {
                                'problem': '_Parity3',
                                'get_data': [
                                    [[0.0, 0.0, 0.0], [0.0]],
                                    [[0.0, 0.0, 1.0], [1.0]],
                                    [[0.0, 1.0, 0.0], [1.0]],
                                    [[0.0, 1.0, 1.0], [0.0]],
                                    [[1.0, 0.0, 0.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [0.0]],
                                    [[1.0, 1.0, 0.0], [0.0]],
                                    [[1.0, 1.0, 1.0], [1.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_coords': [[-1.0, -1.0], [0.0, -1.0], [1.0, -1.0]],
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 0.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 0.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 0.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_coords': [[0.0, 1.0]],
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0], [1.0], [0.0], [0.0], [1.0]],
                            },
                            42,
                            3,
                        ],
                        'value': {
                            'tuple': [
                                [
                                    {'float': '0x1.7bd1e80000000p-1'},
                                    {'float': '0x1.7bd1e80000000p-1'},
                                    {'float': '0x1.7bd1e80000000p-1'},
                                ],
                                {
                                    'array': '<u4',
                                    'shape': [2],
                                    'zlib64': 'eNoT3P9NvnyWxGkAEM0D2g==',
                                },
                            ],
                        },
                    },
                },
            },
            'x86_64': {
                'research': '7440581b1',
                'recorded': '2026-10-23',
                'jax': '0.6.1',
                'jaxlib': '0.6.1',
                'numpy': '1.26.4',
                'xla_flags': ['', '--xla_cpu_multi_thread_eigen=false'],
                'values': {
                    '66596304bcad6d7cc82a259902592c6027df212e9ee19b369020c37ea48cbcc2': {
                        'tests': [
                            'test_hmr_reference.py::test_emr_reproduces_hmrs_trajectory[sin]',
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 60,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'variance_rule': 'legacy',
                                            'max_depth': 2,
                                            'variance_threshold': 0.03,
                                            'dynamic_functions': {
                                                'mode': 'global',
                                                'hidden_activation': 'sin',
                                            },
                                            'extra_randkey_split': True,
                                        },
                                    },
                                },
                            },
                            {
                                'problem': '_Parity3',
                                'get_data': [
                                    [[0.0, 0.0, 0.0], [0.0]],
                                    [[0.0, 0.0, 1.0], [1.0]],
                                    [[0.0, 1.0, 0.0], [1.0]],
                                    [[0.0, 1.0, 1.0], [0.0]],
                                    [[1.0, 0.0, 0.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [0.0]],
                                    [[1.0, 1.0, 0.0], [0.0]],
                                    [[1.0, 1.0, 1.0], [1.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_coords': [[-1.0, -1.0], [0.0, -1.0], [1.0, -1.0]],
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 0.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 0.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 0.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_coords': [[0.0, 1.0]],
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0], [1.0], [0.0], [0.0], [1.0]],
                            },
                            42,
                            3,
                        ],
                        'value': {
                            'tuple': [
                                [
                                    {'float': '0x1.7fb1540000000p-1'},
                                    {'float': '0x1.8685ec0000000p-1'},
                                    {'float': '0x1.ea37440000000p-1'},
                                ],
                                {
                                    'array': '<u4',
                                    'shape': [2],
                                    'zlib64': 'eNoT3P9NvnyWxGkAEM0D2g==',
                                },
                            ],
                        },
                    },
                    'f7bfa31137fdcef710f2c6c8daf1a3ab69a227b2e71533b277ba9cd7b8e8009c': {
                        'tests': [
                            'test_hmr_reference.py::test_emr_reproduces_hmrs_trajectory[tanh]',
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 60,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'variance_rule': 'legacy',
                                            'max_depth': 2,
                                            'variance_threshold': 0.03,
                                            'dynamic_functions': {
                                                'mode': 'global',
                                                'hidden_activation': 'tanh',
                                            },
                                            'extra_randkey_split': True,
                                        },
                                    },
                                },
                            },
                            {
                                'problem': '_Parity3',
                                'get_data': [
                                    [[0.0, 0.0, 0.0], [0.0]],
                                    [[0.0, 0.0, 1.0], [1.0]],
                                    [[0.0, 1.0, 0.0], [1.0]],
                                    [[0.0, 1.0, 1.0], [0.0]],
                                    [[1.0, 0.0, 0.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [0.0]],
                                    [[1.0, 1.0, 0.0], [0.0]],
                                    [[1.0, 1.0, 1.0], [1.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_coords': [[-1.0, -1.0], [0.0, -1.0], [1.0, -1.0]],
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 0.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 0.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 0.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_coords': [[0.0, 1.0]],
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0], [1.0], [0.0], [0.0], [1.0]],
                            },
                            42,
                            3,
                        ],
                        'value': {
                            'tuple': [
                                [
                                    {'float': '0x1.7bd1e80000000p-1'},
                                    {'float': '0x1.7bd1e80000000p-1'},
                                    {'float': '0x1.7bd1e80000000p-1'},
                                ],
                                {
                                    'array': '<u4',
                                    'shape': [2],
                                    'zlib64': 'eNoT3P9NvnyWxGkAEM0D2g==',
                                },
                            ],
                        },
                    },
                },
            },
        },
        'agree': True,
    },
    'leo_hmr_compound.leo_variant': {
        'platforms': {
            'arm64': {
                'research': '7440581b174be13887db6c391e45efc9c3ee2668',
                'recorded': '2026-10-22',
                'jax': '0.6.1',
                'jaxlib': '0.6.1',
                'numpy': '1.26.4',
                'xla_flags': ['', '--xla_cpu_multi_thread_eigen=false'],
                'values': {
                    '093cdc020d6eff11d806625d46cb535e7abfa5ee09f268f9bdea3348230daff0': {
                        'tests': [
                            (
                                'test_leo_hmr_compound.py::TestHmrsLeoVariant::'
                                'test_its_runners_got_a_population_without_the_leo_output'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 12,
                                        'substrate': {
                                            'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'max_depth': 3,
                                            'variance_threshold': 0.03,
                                        },
                                        'neat_species': {
                                            'compatibility_threshold': 2.5,
                                            'max_stagnation': 40,
                                        },
                                    },
                                },
                            },
                            {
                                'problem': '_TwoInputXOR',
                                'get_data': [
                                    [[0.0, 0.0], [0.0]],
                                    [[0.0, 1.0], [1.0]],
                                    [[1.0, 0.0], [1.0]],
                                    [[1.0, 1.0], [0.0]],
                                ],
                                'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                                'input_shape': [2],
                                'inputs': [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
                                'jitable': True,
                                'output_coords': [[0.0, 1.0]],
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                            },
                            42,
                        ],
                        'value': {
                            'dict': [
                                ['leo_output', 6],
                                ['nonzero_weights', 0],
                                [
                                    'population_keys',
                                    {
                                        'array': '<f4',
                                        'shape': [12, 50],
                                        'zlib64': (
                                            'eNpjYACBBnsg4QBEQNwAxAuA+ED90MKj/hj1x6g/Rv0x6o9Rf4z6Y9'
                                            'QfgwMDAPBst4M='
                                        ),
                                    },
                                ],
                                ['rebuilt_population', 150],
                            ],
                        },
                    },
                },
            },
            'x86_64': {
                'research': '7440581b1',
                'recorded': '2026-10-23',
                'jax': '0.6.1',
                'jaxlib': '0.6.1',
                'numpy': '1.26.4',
                'xla_flags': ['', '--xla_cpu_multi_thread_eigen=false'],
                'values': {
                    '093cdc020d6eff11d806625d46cb535e7abfa5ee09f268f9bdea3348230daff0': {
                        'tests': [
                            (
                                'test_leo_hmr_compound.py::TestHmrsLeoVariant::'
                                'test_its_runners_got_a_population_without_the_leo_output'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 12,
                                        'substrate': {
                                            'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'max_depth': 3,
                                            'variance_threshold': 0.03,
                                        },
                                        'neat_species': {
                                            'compatibility_threshold': 2.5,
                                            'max_stagnation': 40,
                                        },
                                    },
                                },
                            },
                            {
                                'problem': '_TwoInputXOR',
                                'get_data': [
                                    [[0.0, 0.0], [0.0]],
                                    [[0.0, 1.0], [1.0]],
                                    [[1.0, 0.0], [1.0]],
                                    [[1.0, 1.0], [0.0]],
                                ],
                                'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                                'input_shape': [2],
                                'inputs': [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
                                'jitable': True,
                                'output_coords': [[0.0, 1.0]],
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                            },
                            42,
                        ],
                        'value': {
                            'dict': [
                                ['leo_output', 6],
                                ['nonzero_weights', 0],
                                [
                                    'population_keys',
                                    {
                                        'array': '<f4',
                                        'shape': [12, 50],
                                        'zlib64': (
                                            'eNpjYACBBnsg4QBEQNwAxAuA+ED90MKj/hj1x6g/Rv0x6o9Rf4z6Y9'
                                            'QfgwMDAPBst4M='
                                        ),
                                    },
                                ],
                                ['rebuilt_population', 150],
                            ],
                        },
                    },
                },
            },
        },
        'agree': True,
    },
    'live_palette_strategy.hmr_perturbation': {
        'platforms': {
            'arm64': {
                'research': '7440581b174be13887db6c391e45efc9c3ee2668',
                'recorded': '2026-10-22',
                'jax': '0.6.1',
                'jaxlib': '0.6.1',
                'numpy': '1.26.4',
                'xla_flags': ['', '--xla_cpu_multi_thread_eigen=false'],
                'values': {
                    'a64a69614323258bfa9ee50590f2c91d25c26714203e9a2b760dd02199a1b675': {
                        'tests': [
                            (
                                'test_live_palette_strategy.py::TestPerturbation::'
                                'test_negative_control_hmrs_function'
                            ),
                        ],
                        'inputs': ['<f4[20, 50, 5]', '<u4[2]', [0.25, 0.3, 0.0, 1, 4, 5]],
                        'value': {
                            'tuple': [
                                {
                                    'array': '<f4',
                                    'shape': [20, 50, 5],
                                    'zlib64': (
                                        'eNrtnG1IVFkYx69hFLVUJBW7E41DmYm1sWhR7cw9T1uLFmqmYS+bWC1qL4'
                                        'ZQuWhFTrltWWRmrzNEjSnVkMtWG1uic8+IvaBl2pZCJQjZsttCsB9aQSmq'
                                        'aWHx/s/St7n75RF+KJc744fhN//nOec5V9M07UH8dl3T3EL798ctzqy6Nu'
                                        'jah980vTCSzPcRrT71PGC+z031Y2bC+/noTsxX0nxfkL41Ngi4VsowTDjQ'
                                        'NHlr0iAHtQ8uB4sewjWNikU/uE/0+niNgNeStm4hOO2juM8OgftBWtOQDv'
                                        '+DPw+GCZfnFTfaBOb5+K6VAczzLdF2xfOkc0MVz7/cvkPJ81XOxYrnruzL'
                                        'LvacYcKP1+PRL9reDvZNy8s7Lx4V1ptq6oSEKCqy3zcG35eQkEndSwZM17'
                                        'yecmr52mHKc6/HT11BlzBfa6Evxs5C9/kzYZgwed5U/bOOnm/uX2HK6ZDn'
                                        '/vxhLvS8oDhON/tbTnM7PAZ6njVmh0TPe6ekGZznDGNN3Z5sS8TaW7TXJE'
                                        'tz1mo0e8GbANbtJSkBietwk0ucBtbtRlUA19xoxJ4t3J8zjEWeu9c3NICr'
                                        'otCxW6DnE/3TgmYviR74Cw30fOEen0DPi7KbJPbn9b/uZc8ZxqK6vTN6lM'
                                        'C6fc254RL78wt/ThNYtz9rLzC5GqrbuzLSdXPP7qc/yuuUuj3rUpXk/pxh'
                                        'rMnzhwdjlLp968k2gevtIw80Cdw/nzc8Q6nbN87wuzDP6+S9AOb5d2XxxH'
                                        'nOMBbtny86rczJVN6JVTxfXrlRouetFdnQi7vpZsQn8L3ho7NTtir7ahE9'
                                        'vxg8J8Mw1ni+NyJH8fz2q/2K50O7rxvoeX/SWYF5PiH/Nx3nZBJT7NiLU9'
                                        '7hJzwPxzAW9efjN9eavAz156XBeOjFo+jzqdWGea8tk0aXdUvcV8tu/d1l'
                                        '7tn9FFsRo/TnsUNOCe7PGcaaPO8/0KPMvT7+oUPg3OsmZ6TAfbXKvmYD5+'
                                        'GWRl9R8nzcp/lKf55W7WDPGcaiPE/svigwz5P+8ho4J9Nni4MZuUzqSe80'
                                        'MM9XbEgVmOc1teWwBt9CzWU7JdftDGNNnh+9tVaZb4+3eZ2Y5zcv3ZW4f/'
                                        '53ca6Oed44IujCPK+K61X688rcHN4/ZxiLPB+4blc8P3E+zcB1uKe7BnRc'
                                        'hzt0tU6Zk/mvcyyvFzmUc6kw3/7e88vzZ0W18cw7w4TB8/Xf9CqeF+2bKd'
                                        'Fze7JHYn+O51hCnve9tCnnUn/qrNUxz/kcC8NY5zmeVwv9LVe+Ujx3pJYq'
                                        'nmfkLFbmZOZk7ddx/9x37Ed+zgTD/I+erxsdJdHz5x23dezPm9PsyvNkuv'
                                        'K+d6LnL5b1KudYZkQWiI8/Z4I9Z5hwer5cvlTmZPD8eUjqk/sWKP15yZFI'
                                        'ZV+tsWNb48efM/FPnrfPfabzOhzDhJ93Ac4LLA=='
                                    ),
                                },
                                {
                                    'array': '<u4',
                                    'shape': [2],
                                    'zlib64': 'eNozajdcG8l49iYADaoDmA==',
                                },
                            ],
                        },
                    },
                },
            },
            'x86_64': {
                'research': '7440581b1',
                'recorded': '2026-10-23',
                'jax': '0.6.1',
                'jaxlib': '0.6.1',
                'numpy': '1.26.4',
                'xla_flags': ['', '--xla_cpu_multi_thread_eigen=false'],
                'values': {
                    'a64a69614323258bfa9ee50590f2c91d25c26714203e9a2b760dd02199a1b675': {
                        'tests': [
                            (
                                'test_live_palette_strategy.py::TestPerturbation::'
                                'test_negative_control_hmrs_function'
                            ),
                        ],
                        'inputs': ['<f4[20, 50, 5]', '<u4[2]', [0.25, 0.3, 0.0, 1, 4, 5]],
                        'value': {
                            'tuple': [
                                {
                                    'array': '<f4',
                                    'shape': [20, 50, 5],
                                    'zlib64': (
                                        'eNrtnG1IVFkYx69hFLVUJBW7E41DmYm1sWhR7cw9T1uLFmqmYS+bWC1qL4'
                                        'ZQuWhFTrltWWRmrzNEjSnVkMtWG1uic8+IvaBl2pZCJQjZsttCsB9aQSmq'
                                        'aWHx/s/St7n75RF+KJc744fhN//nOec5V9M07UH8dl3T3EL798ctzqy6Nu'
                                        'jah980vTCSzPcRrT71PGC+z031Y2bC+/noTsxX0nxfkL41Ngi4VsowTDjQ'
                                        'NHlr0iAHtQ8uB4sewjWNikU/uE/0+niNgNeStm4hOO2juM8OgftBWtOQDv'
                                        '+DPw+GCZfnFTfaBOb5+K6VAczzLdF2xfOkc0MVz7/cvkPJ81XOxYrnruzL'
                                        'LvacYcKP1+PRL9reDvZNy8s7Lx4V1ptq6oSEKCqy3zcG35eQkEndSwZM17'
                                        'yecmr52mHKc6/HT11BlzBfa6Evxs5C9/kzYZgwed5U/bOOnm/uX2HK6ZDn'
                                        '/vxhLvS8oDhON/tbTnM7PAZ6njVmh0TPe6ekGZznDGNN3Z5sS8TaW7TXJE'
                                        'tz1mo0e8GbANbtJSkBietwk0ucBtbtRlUA19xoxJ4t3J8zjEWeu9c3NICr'
                                        'otCxW6DnE/3TgmYviR74Cw30fOEen0DPi7KbJPbn9b/uZc8ZxqK6vTN6lM'
                                        'C6fc254RL78wt/ThNYtz9rLzC5GqrbuzLSdXPP7qc/yuuUuj3rUpXk/pxh'
                                        'rMnzhwdjlLp968k2gevtIw80Cdw/nzc8Q6nbN87wuzDP6+S9AOb5d2XxxH'
                                        'nOMBbtny86rczJVN6JVTxfXrlRouetFdnQi7vpZsQn8L3ho7NTtir7ahE9'
                                        'vxg8J8Mw1ni+NyJH8fz2q/2K50O7rxvoeX/SWYF5PiH/Nx3nZBJT7NiLU9'
                                        '7hJzwPxzAW9efjN9eavAz156XBeOjFo+jzqdWGea8tk0aXdUvcV8tu/d1l'
                                        '7tn9FFsRo/TnsUNOCe7PGcaaPO8/0KPMvT7+oUPg3OsmZ6TAfbXKvmYD5+'
                                        'GWRl9R8nzcp/lKf55W7WDPGcaiPE/svigwz5P+8ho4J9Nni4MZuUzqSe80'
                                        'MM9XbEgVmOc1teWwBt9CzWU7JdftDGNNnh+9tVaZb4+3eZ2Y5zcv3ZW4f/'
                                        '53ca6Oed44IujCPK+K61X688rcHN4/ZxiLPB+4blc8P3E+zcB1uKe7BnRc'
                                        'hzt0tU6Zk/mvcyyvFzmUc6kw3/7e88vzZ0W18cw7w4TB8/Xf9CqeF+2bKd'
                                        'Fze7JHYn+O51hCnve9tCnnUn/qrNUxz/kcC8NY5zmeVwv9LVe+Ujx3pJYq'
                                        'nmfkLFbmZOZk7ddx/9x37Ed+zgTD/I+erxsdJdHz5x23dezPm9PsyvNkuv'
                                        'K+d6LnL5b1KudYZkQWiI8/Z4I9Z5hwer5cvlTmZPD8eUjqk/sWKP15yZFI'
                                        'ZV+tsWNb48efM/FPnrfPfabzOhzDhJ93Ac4LLA=='
                                    ),
                                },
                                {
                                    'array': '<u4',
                                    'shape': [2],
                                    'zlib64': 'eNozajdcG8l49iYADaoDmA==',
                                },
                            ],
                        },
                    },
                },
            },
        },
        'agree': True,
    },
    'node_list.gr_dense': {
        'platforms': {
            'arm64': {
                'research': '7440581b174be13887db6c391e45efc9c3ee2668',
                'recorded': '2026-10-22',
                'jax': '0.6.1',
                'jaxlib': '0.6.1',
                'numpy': '1.26.4',
                'xla_flags': ['', '--xla_cpu_multi_thread_eigen=false'],
                'values': {
                    '6c21e5931e807700ae42f29e915b753e1ff45e83394242d42a8258a36a9dbfd2': {
                        'tests': [
                            'test_node_list.py::TestGRRecipe::test_configured_band_is_not_grs',
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'max_depth': 2,
                                            'variance_threshold': 0.0,
                                            'band_threshold': 0.3,
                                            'max_weight': 8.0,
                                            'recurrence': {'preset': 'full_recurrent'},
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            2,
                        ],
                        'value': {
                            'dict': [
                                [
                                    'fitness',
                                    [
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': 'eNpjYGCwZ8CCWXCIE8IAwSQE8Q==',
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': 'eNpjYWCwZ6AiBgDBhATx',
                                        },
                                    ],
                                ],
                                ['methods', ['dense_recurrent', 'dense_recurrent']],
                            ],
                        },
                    },
                    '75ca0ddbbdd5def792be3266ee62d53cb530745d99c2510b4a2f251b88abc91d': {
                        'tests': [
                            (
                                'test_node_list.py::TestGRRecipe::'
                                'test_depths_three_and_four[3-0.03-20]'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'max_depth': 3,
                                            'variance_threshold': 0.03,
                                            'band_threshold': 0.3,
                                            'max_weight': 8.0,
                                            'recurrence': {'preset': 'full_recurrent'},
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            2,
                        ],
                        'value': {
                            'dict': [
                                [
                                    'fitness',
                                    [
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNqr+Vdvx8DAYN9R8gNMw3CRKYKNDQtxN9htOboNricihgtMAw'
                                                'BGtQ0H'
                                            ),
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': 'eNqLiOGyLzJlsGdgIA8rsKDyAS7eBnY=',
                                        },
                                    ],
                                ],
                                ['methods', ['dense_recurrent', 'dense_recurrent']],
                            ],
                        },
                    },
                    '8d7e1c56cbea471cf0aa948b7fd28a3e25cae9a1fde89b485b5e14ce70a84779': {
                        'tests': [
                            (
                                'test_node_list.py::TestGRRecipe::'
                                'test_depths_three_and_four[4-0.03-20]'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'max_depth': 4,
                                            'variance_threshold': 0.03,
                                            'band_threshold': 0.3,
                                            'max_weight': 8.0,
                                            'recurrence': {'preset': 'full_recurrent'},
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            2,
                        ],
                        'value': {
                            'dict': [
                                [
                                    'fitness',
                                    [
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNqr+Vdvx8DAYN9R8gNMw3CRKYKNDQtxN9htOboNricihgtMAw'
                                                'BGtQ0H'
                                            ),
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': 'eNqLiOGyLzJlsGdgIA8rsKDyAS7eBnY=',
                                        },
                                    ],
                                ],
                                ['methods', ['dense_recurrent', 'dense_recurrent']],
                            ],
                        },
                    },
                    'c3e4608c2a69744af6850ee5b1b6e279e6a4e077b427a0abb01c080a01bfeb63': {
                        'tests': [
                            'test_node_list.py::TestGRRecipe::test_depths_three_and_four[3-0.0-20]',
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'max_depth': 3,
                                            'variance_threshold': 0.0,
                                            'band_threshold': 0.3,
                                            'max_weight': 8.0,
                                            'recurrence': {'preset': 'full_recurrent'},
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            2,
                        ],
                        'value': {
                            'dict': [
                                [
                                    'fitness',
                                    [
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': 'eNpjYGCwZ6AiBgDARATt',
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': 'eNpjYGCwZ6AiBgDARATt',
                                        },
                                    ],
                                ],
                                ['methods', ['dense_recurrent', 'dense_recurrent']],
                            ],
                        },
                    },
                    'f15a67bc4a1d79703d207feae749a98b03d698199833f2d86fe0cdefb3380f61': {
                        'tests': ['test_node_list.py::TestGRRecipe::test_depth_two[0.03]'],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'max_depth': 2,
                                            'variance_threshold': 0.03,
                                            'band_threshold': 0.3,
                                            'max_weight': 8.0,
                                            'recurrence': {'preset': 'full_recurrent'},
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            3,
                        ],
                        'value': {
                            'dict': [
                                [
                                    'fitness',
                                    [
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNqr+Vdvx8DAYN9R8gNMw3CRKYKNDQtxN9htOboNricihgtMAw'
                                                'BGtQ0H'
                                            ),
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': 'eNqLiOGyLzJlsGdgIA8rsKDyAS7eBnY=',
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNqLiOGyLzJlsFdgYbBnYCANB61mtd8vrgTni7Az2AMAWbsIiw'
                                                '=='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'methods',
                                    ['dense_recurrent', 'dense_recurrent', 'dense_recurrent'],
                                ],
                            ],
                        },
                    },
                    'f8e36165a01a7fe0f9209e884f60a532c00e2ac26930381b846a0bed2a18f7c3': {
                        'tests': ['test_node_list.py::TestGRRecipe::test_emr_cap_is_not_grs'],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 10,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'max_depth': 3,
                                            'variance_threshold': 0.0,
                                            'band_threshold': 0.3,
                                            'max_weight': 8.0,
                                            'recurrence': {'preset': 'full_recurrent'},
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            2,
                        ],
                        'value': {
                            'dict': [
                                [
                                    'fitness',
                                    [
                                        {
                                            'array': '<f4',
                                            'shape': [10],
                                            'zlib64': 'eNpjYGCwZyACAwAu6gJ3',
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [10],
                                            'zlib64': 'eNpjYGCwZyACAwAu6gJ3',
                                        },
                                    ],
                                ],
                                ['methods', ['dense_recurrent', 'dense_recurrent']],
                            ],
                        },
                    },
                    'fabadd09cdf31922ba22ba43e2ad3eb35ff7217c10dcc3631dca0b6198976dba': {
                        'tests': ['test_node_list.py::TestGRRecipe::test_depth_two[0.0]'],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'max_depth': 2,
                                            'variance_threshold': 0.0,
                                            'band_threshold': 0.3,
                                            'max_weight': 8.0,
                                            'recurrence': {'preset': 'full_recurrent'},
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            3,
                        ],
                        'value': {
                            'dict': [
                                [
                                    'fitness',
                                    [
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': 'eNpjYGCwZ8CCWXCIE8IAwSQE8Q==',
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': 'eNpjYWCwZ6AiBgDBhATx',
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': 'eNpjYWCwZyATMzEYoIk52AMAxYwFYw==',
                                        },
                                    ],
                                ],
                                [
                                    'methods',
                                    ['dense_recurrent', 'dense_recurrent', 'dense_recurrent'],
                                ],
                            ],
                        },
                    },
                },
            },
            'x86_64': {
                'research': '7440581b1',
                'recorded': '2026-10-23',
                'jax': '0.6.1',
                'jaxlib': '0.6.1',
                'numpy': '1.26.4',
                'xla_flags': ['', '--xla_cpu_multi_thread_eigen=false'],
                'values': {
                    '6c21e5931e807700ae42f29e915b753e1ff45e83394242d42a8258a36a9dbfd2': {
                        'tests': [
                            'test_node_list.py::TestGRRecipe::test_configured_band_is_not_grs',
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'max_depth': 2,
                                            'variance_threshold': 0.0,
                                            'band_threshold': 0.3,
                                            'max_weight': 8.0,
                                            'recurrence': {'preset': 'full_recurrent'},
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            2,
                        ],
                        'value': {
                            'dict': [
                                [
                                    'fitness',
                                    [
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': 'eNpjYGCwZ8CCWXCIE8IAwSQE8Q==',
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': 'eNpjYWCwZ6AiBgDBhATx',
                                        },
                                    ],
                                ],
                                ['methods', ['dense_recurrent', 'dense_recurrent']],
                            ],
                        },
                    },
                    '75ca0ddbbdd5def792be3266ee62d53cb530745d99c2510b4a2f251b88abc91d': {
                        'tests': [
                            (
                                'test_node_list.py::TestGRRecipe::'
                                'test_depths_three_and_four[3-0.03-20]'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'max_depth': 3,
                                            'variance_threshold': 0.03,
                                            'band_threshold': 0.3,
                                            'max_weight': 8.0,
                                            'recurrence': {'preset': 'full_recurrent'},
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            2,
                        ],
                        'value': {
                            'dict': [
                                [
                                    'fitness',
                                    [
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNqr+Vdvx8DAYN9R8gNMw3CRKYKNDQtxN9htOboNricihgtMAw'
                                                'BGtQ0H'
                                            ),
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': 'eNqLiOGyLzJlsGdgIA8rsKDyAS7eBnY=',
                                        },
                                    ],
                                ],
                                ['methods', ['dense_recurrent', 'dense_recurrent']],
                            ],
                        },
                    },
                    '8d7e1c56cbea471cf0aa948b7fd28a3e25cae9a1fde89b485b5e14ce70a84779': {
                        'tests': [
                            (
                                'test_node_list.py::TestGRRecipe::'
                                'test_depths_three_and_four[4-0.03-20]'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'max_depth': 4,
                                            'variance_threshold': 0.03,
                                            'band_threshold': 0.3,
                                            'max_weight': 8.0,
                                            'recurrence': {'preset': 'full_recurrent'},
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            2,
                        ],
                        'value': {
                            'dict': [
                                [
                                    'fitness',
                                    [
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNqr+Vdvx8DAYN9R8gNMw3CRKYKNDQtxN9htOboNricihgtMAw'
                                                'BGtQ0H'
                                            ),
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': 'eNqLiOGyLzJlsGdgIA8rsKDyAS7eBnY=',
                                        },
                                    ],
                                ],
                                ['methods', ['dense_recurrent', 'dense_recurrent']],
                            ],
                        },
                    },
                    'c3e4608c2a69744af6850ee5b1b6e279e6a4e077b427a0abb01c080a01bfeb63': {
                        'tests': [
                            'test_node_list.py::TestGRRecipe::test_depths_three_and_four[3-0.0-20]',
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'max_depth': 3,
                                            'variance_threshold': 0.0,
                                            'band_threshold': 0.3,
                                            'max_weight': 8.0,
                                            'recurrence': {'preset': 'full_recurrent'},
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            2,
                        ],
                        'value': {
                            'dict': [
                                [
                                    'fitness',
                                    [
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': 'eNpjYGCwZ6AiBgDARATt',
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': 'eNpjYGCwZ6AiBgDARATt',
                                        },
                                    ],
                                ],
                                ['methods', ['dense_recurrent', 'dense_recurrent']],
                            ],
                        },
                    },
                    'f15a67bc4a1d79703d207feae749a98b03d698199833f2d86fe0cdefb3380f61': {
                        'tests': ['test_node_list.py::TestGRRecipe::test_depth_two[0.03]'],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'max_depth': 2,
                                            'variance_threshold': 0.03,
                                            'band_threshold': 0.3,
                                            'max_weight': 8.0,
                                            'recurrence': {'preset': 'full_recurrent'},
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            3,
                        ],
                        'value': {
                            'dict': [
                                [
                                    'fitness',
                                    [
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNqr+Vdvx8DAYN9R8gNMw3CRKYKNDQtxN9htOboNricihgtMAw'
                                                'BGtQ0H'
                                            ),
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': 'eNqLiOGyLzJlsGdgIA8rsKDyAS7eBnY=',
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNqLiOGyLzJlsFdgYbBnYCANB61mtd8vrgTni7Az2AMAWbsIiw'
                                                '=='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'methods',
                                    ['dense_recurrent', 'dense_recurrent', 'dense_recurrent'],
                                ],
                            ],
                        },
                    },
                    'f8e36165a01a7fe0f9209e884f60a532c00e2ac26930381b846a0bed2a18f7c3': {
                        'tests': ['test_node_list.py::TestGRRecipe::test_emr_cap_is_not_grs'],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 10,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'max_depth': 3,
                                            'variance_threshold': 0.0,
                                            'band_threshold': 0.3,
                                            'max_weight': 8.0,
                                            'recurrence': {'preset': 'full_recurrent'},
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            2,
                        ],
                        'value': {
                            'dict': [
                                [
                                    'fitness',
                                    [
                                        {
                                            'array': '<f4',
                                            'shape': [10],
                                            'zlib64': 'eNpjYGCwZyACAwAu6gJ3',
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [10],
                                            'zlib64': 'eNpjYGCwZyACAwAu6gJ3',
                                        },
                                    ],
                                ],
                                ['methods', ['dense_recurrent', 'dense_recurrent']],
                            ],
                        },
                    },
                    'fabadd09cdf31922ba22ba43e2ad3eb35ff7217c10dcc3631dca0b6198976dba': {
                        'tests': ['test_node_list.py::TestGRRecipe::test_depth_two[0.0]'],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'max_depth': 2,
                                            'variance_threshold': 0.0,
                                            'band_threshold': 0.3,
                                            'max_weight': 8.0,
                                            'recurrence': {'preset': 'full_recurrent'},
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            3,
                        ],
                        'value': {
                            'dict': [
                                [
                                    'fitness',
                                    [
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': 'eNpjYGCwZ8CCWXCIE8IAwSQE8Q==',
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': 'eNpjYWCwZ6AiBgDBhATx',
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': 'eNpjYWCwZyAT//uvjybmYA8A7BcHXQ==',
                                        },
                                    ],
                                ],
                                [
                                    'methods',
                                    ['dense_recurrent', 'dense_recurrent', 'dense_recurrent'],
                                ],
                            ],
                        },
                    },
                },
            },
        },
        'agree': False,
    },
    'node_list.gr_failure': {
        'platforms': {
            'arm64': {
                'research': '7440581b174be13887db6c391e45efc9c3ee2668',
                'recorded': '2026-10-22',
                'jax': '0.6.1',
                'jaxlib': '0.6.1',
                'numpy': '1.26.4',
                'xla_flags': ['', '--xla_cpu_multi_thread_eigen=false'],
                'values': {
                    'c1e61c6be8cedbdda74a366817c7f008aeada0f11e18e76d842e81cebb60a04f': {
                        'tests': [
                            'test_node_list.py::TestGRRecipe::test_gr_fails_without_full_recurrent',
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 8,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'max_depth': 2,
                                            'variance_threshold': 0.03,
                                            'band_threshold': 0.3,
                                            'max_weight': 8.0,
                                            'recurrence': {'preset': 'hidden_only'},
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            1,
                        ],
                        'value': {
                            'dict': [
                                ['error', 'ValueError'],
                                [
                                    'message',
                                    'scan got no values to scan over and `length` not provided.',
                                ],
                            ],
                        },
                    },
                },
            },
            'x86_64': {
                'research': '7440581b1',
                'recorded': '2026-10-23',
                'jax': '0.6.1',
                'jaxlib': '0.6.1',
                'numpy': '1.26.4',
                'xla_flags': ['', '--xla_cpu_multi_thread_eigen=false'],
                'values': {
                    'c1e61c6be8cedbdda74a366817c7f008aeada0f11e18e76d842e81cebb60a04f': {
                        'tests': [
                            'test_node_list.py::TestGRRecipe::test_gr_fails_without_full_recurrent',
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 8,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'max_depth': 2,
                                            'variance_threshold': 0.03,
                                            'band_threshold': 0.3,
                                            'max_weight': 8.0,
                                            'recurrence': {'preset': 'hidden_only'},
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            1,
                        ],
                        'value': {
                            'dict': [
                                ['error', 'ValueError'],
                                [
                                    'message',
                                    'scan got no values to scan over and `length` not provided.',
                                ],
                            ],
                        },
                    },
                },
            },
        },
        'agree': True,
    },
    'node_list.rn_chain_keys': {
        'platforms': {
            'arm64': {
                'research': '7440581b174be13887db6c391e45efc9c3ee2668',
                'recorded': '2026-10-22',
                'jax': '0.6.1',
                'jaxlib': '0.6.1',
                'numpy': '1.26.4',
                'xla_flags': ['', '--xla_cpu_multi_thread_eigen=false'],
                'values': {
                    'defd62bb47da3670a436957d82af89cf30527a772bd7dcbc0fcd045e13ea5bd3': {
                        'tests': ['test_node_list.py::TestChecks::test_depth_six_keys'],
                        'inputs': [[[-1.0, -1.0], [0.0, -1.0], [1.0, -1.0]], [[0.0, 1.0]], [5, 6]],
                        'value': [3, 0],
                    },
                },
            },
            'x86_64': {
                'research': '7440581b1',
                'recorded': '2026-10-23',
                'jax': '0.6.1',
                'jaxlib': '0.6.1',
                'numpy': '1.26.4',
                'xla_flags': ['', '--xla_cpu_multi_thread_eigen=false'],
                'values': {
                    'defd62bb47da3670a436957d82af89cf30527a772bd7dcbc0fcd045e13ea5bd3': {
                        'tests': ['test_node_list.py::TestChecks::test_depth_six_keys'],
                        'inputs': [[[-1.0, -1.0], [0.0, -1.0], [1.0, -1.0]], [[0.0, 1.0]], [5, 6]],
                        'value': [3, 0],
                    },
                },
            },
        },
        'agree': True,
    },
    'node_list.rn_cleaning': {
        'platforms': {
            'arm64': {
                'research': '7440581b174be13887db6c391e45efc9c3ee2668',
                'recorded': '2026-10-22',
                'jax': '0.6.1',
                'jaxlib': '0.6.1',
                'numpy': '1.26.4',
                'xla_flags': ['', '--xla_cpu_multi_thread_eigen=false'],
                'values': {
                    '0f96733d85e18153c3c42f44329de5adedc27a59512f4de99890d5dbed631300': {
                        'tests': ['test_node_list.py::TestPieces::test_cleaning_is_rns[4]'],
                        'inputs': ['<f4[9, 5]', '<f4[3, 2]', '<f4[1, 2]'],
                        'value': {
                            'array': '<f4',
                            'shape': [6, 5],
                            'zlib64': (
                                'eNpjYGjYzwDGDiBsD2QDMQiAxfchxGBsMN8OIgZn2yHUwcwBmwHTa4cuBgBvQhUg'
                            ),
                        },
                    },
                    '28afb60314c0a0a51bc0b5bcaa95d267de9426715ea48e6e865c1559f72111bf': {
                        'tests': ['test_node_list.py::TestPieces::test_cleaning_is_rns[1]'],
                        'inputs': ['<f4[27, 5]', '<f4[3, 2]', '<f4[1, 2]'],
                        'value': {
                            'array': '<f4',
                            'shape': [18, 5],
                            'zlib64': (
                                'eNpdjwEOABAMA/cyPM3L+BornZpkkZ21OrM+DdX8LruqBUM/DvPDOcwqq6ec8R39eC'
                                'y8Estz9KIfdV++crXC6KP5wlv2+Jhom/yvTOeYH3unfMqofWwBfw5H0A=='
                            ),
                        },
                    },
                    '2cef9a9fafff669d1e72982d5c218e3429164ddea2a442817e69e0c31ad21330': {
                        'tests': ['test_node_list.py::TestPieces::test_cleaning_is_rns[5]'],
                        'inputs': ['<f4[38, 5]', '<f4[3, 2]', '<f4[1, 2]'],
                        'value': {
                            'array': '<f4',
                            'shape': [36, 5],
                            'zlib64': (
                                'eNp1kQEOgDAIA3kZ5Wm+TL9m0IC0myaL4VJaxsyOy95zmkX+YcwgzD9dfsmyLl2x2d'
                                's6vFxZZ+DrzZys4+KjjHTgWWL4d8a5+nWes67qqQtIrw8/MKOdlrf/MDArfmDMMHba'
                                'bPiF3I+Y+mHjJ7sqn+etBkufySpT2VYHYb7p9VVX8y6MMm44e4oz'
                            ),
                        },
                    },
                    '3afb8c533e195b26221b4ee1a27d6b729bfc8079f158c7539629fe4a09e1b65d': {
                        'tests': ['test_node_list.py::TestPieces::test_cleaning_is_rns[7]'],
                        'inputs': ['<f4[26, 5]', '<f4[3, 2]', '<f4[1, 2]'],
                        'value': {
                            'array': '<f4',
                            'shape': [13, 5],
                            'zlib64': (
                                'eNpljwEKACAIA/cy19N6WX4tSzQrQczbhgSs6mo9gEabxGFiTJ3tVt8/RveuPfVRfP'
                                'GWh/HOpi6uB4t8ZkMvN9rjrXf3f3j7DpsDRzZw'
                            ),
                        },
                    },
                    '3dd6cbfb2425dc5ec2c2f10ab4f7865c6f06316ac9ccfa40fa21fd2be8d1d05d': {
                        'tests': ['test_node_list.py::TestPieces::test_cleaning_is_rns[3]'],
                        'inputs': ['<f4[17, 5]', '<f4[3, 2]', '<f4[1, 2]'],
                        'value': {
                            'array': '<f4',
                            'shape': [15, 5],
                            'zlib64': (
                                'eNp1zsENACAIA0Amo4zGZLiaUUHBqEk/FwoSaaMZ8Shom9rDePmw8e65MMHH4PvwMD'
                                'v/2Bb32IPUSXejN28hGapFp5jvLl3ccx3P+zsx'
                            ),
                        },
                    },
                    'a0befeb3a4114eb14f3632ffab6a2fa92b609154af95e57475a1eda6248e6a13': {
                        'tests': ['test_node_list.py::TestPieces::test_cleaning_is_rns[6]'],
                        'inputs': ['<f4[31, 5]', '<f4[3, 2]', '<f4[1, 2]'],
                        'value': {
                            'array': '<f4',
                            'shape': [19, 5],
                            'zlib64': (
                                'eNptj4ENACEIA5kMGM3JYDUR0RT/TUjM0RYgGk67OMqihC7T+Ktvtl4y/7L02aMD78'
                                'lcLOcI5ODcylHpLLMwj0EL7OiO93qqN7BvD+Oe15jALl43St+5Mf5nKsgmuQtNzA=='
                            ),
                        },
                    },
                    'a1d9ab854b1ba019681a0f5e1b0f9a012582b6c48c703eea38e29b600ce36531': {
                        'tests': [
                            'test_node_list.py::TestPieces::test_cleaning_reaches_its_fixpoint',
                        ],
                        'inputs': ['<f4[61, 5]', '<f4[3, 2]', '<f4[1, 2]'],
                        'value': {
                            'array': '<f4',
                            'shape': [41, 5],
                            'zlib64': (
                                'eNpNklsNxDAQxAJhoQyBPiAESiEESojclUqg3KXdUfyxGstq+2G1lOsu5dzLs/2/7V'
                                'g8d8CNdHEsF7mCU26Fq7kX3OR6v9zANc/OJzilC7hIN77LTZ7b4Xq6E+5M1z7LTX52'
                                'g9vy+R3v7vlduJ5uwI107meu6Geu6Geu6GcW+pmFfkI7wfkCLtK5n1noZxb6mYV+Zq'
                                'GfWehnFvqZhX5moZ9Z6GcW+pmFfuZAP3OgX+A/e/v9ADWnuTc='
                            ),
                        },
                    },
                    'a7bd8898eeca75d75e7ee3fafca76189841264529c159ed9045c6cb829a46464': {
                        'tests': ['test_node_list.py::TestPieces::test_cleaning_is_rns[2]'],
                        'inputs': ['<f4[22, 5]', '<f4[3, 2]', '<f4[1, 2]'],
                        'value': {
                            'array': '<f4',
                            'shape': [10, 5],
                            'zlib64': (
                                'eNpjYGjYzwDGDiBsD2QDMQg0wPj7IGIgeRAbTNshidlB8T5UMXS9DkhmgMRgelDMs8'
                                'cthmIHVAzsTrCbAQsNKr8='
                            ),
                        },
                    },
                    'e9740ef1621dea9a858a9bcb98260b214701824562a8c80a284e4e361d661731': {
                        'tests': ['test_node_list.py::TestPieces::test_cleaning_is_rns[0]'],
                        'inputs': ['<f4[32, 5]', '<f4[3, 2]', '<f4[1, 2]'],
                        'value': {
                            'array': '<f4',
                            'shape': [19, 5],
                            'zlib64': (
                                'eNplkIENwCAIBJkMGI3J6mpWi3pUk0/kzPOgSDSZ8iF77yabhd5s1k8yg7d9tbcqMv'
                                'ZzZiqYHcV608zUyuglo9fZw8783GNn4g/KHCsjZxgnfhlkq/fFircDKndHzg=='
                            ),
                        },
                    },
                },
            },
            'x86_64': {
                'research': '7440581b1',
                'recorded': '2026-10-23',
                'jax': '0.6.1',
                'jaxlib': '0.6.1',
                'numpy': '1.26.4',
                'xla_flags': ['', '--xla_cpu_multi_thread_eigen=false'],
                'values': {
                    '0f96733d85e18153c3c42f44329de5adedc27a59512f4de99890d5dbed631300': {
                        'tests': ['test_node_list.py::TestPieces::test_cleaning_is_rns[4]'],
                        'inputs': ['<f4[9, 5]', '<f4[3, 2]', '<f4[1, 2]'],
                        'value': {
                            'array': '<f4',
                            'shape': [6, 5],
                            'zlib64': (
                                'eNpjYGjYzwDGDiBsD2QDMQiAxfchxGBsMN8OIgZn2yHUwcwBmwHTa4cuBgBvQhUg'
                            ),
                        },
                    },
                    '28afb60314c0a0a51bc0b5bcaa95d267de9426715ea48e6e865c1559f72111bf': {
                        'tests': ['test_node_list.py::TestPieces::test_cleaning_is_rns[1]'],
                        'inputs': ['<f4[27, 5]', '<f4[3, 2]', '<f4[1, 2]'],
                        'value': {
                            'array': '<f4',
                            'shape': [18, 5],
                            'zlib64': (
                                'eNpdjwEOABAMA/cyPM3L+BornZpkkZ21OrM+DdX8LruqBUM/DvPDOcwqq6ec8R39eC'
                                'y8Estz9KIfdV++crXC6KP5wlv2+Jhom/yvTOeYH3unfMqofWwBfw5H0A=='
                            ),
                        },
                    },
                    '2cef9a9fafff669d1e72982d5c218e3429164ddea2a442817e69e0c31ad21330': {
                        'tests': ['test_node_list.py::TestPieces::test_cleaning_is_rns[5]'],
                        'inputs': ['<f4[38, 5]', '<f4[3, 2]', '<f4[1, 2]'],
                        'value': {
                            'array': '<f4',
                            'shape': [36, 5],
                            'zlib64': (
                                'eNp1kQEOgDAIA3kZ5Wm+TL9m0IC0myaL4VJaxsyOy95zmkX+YcwgzD9dfsmyLl2x2d'
                                's6vFxZZ+DrzZys4+KjjHTgWWL4d8a5+nWes67qqQtIrw8/MKOdlrf/MDArfmDMMHba'
                                'bPiF3I+Y+mHjJ7sqn+etBkufySpT2VYHYb7p9VVX8y6MMm44e4oz'
                            ),
                        },
                    },
                    '3afb8c533e195b26221b4ee1a27d6b729bfc8079f158c7539629fe4a09e1b65d': {
                        'tests': ['test_node_list.py::TestPieces::test_cleaning_is_rns[7]'],
                        'inputs': ['<f4[26, 5]', '<f4[3, 2]', '<f4[1, 2]'],
                        'value': {
                            'array': '<f4',
                            'shape': [13, 5],
                            'zlib64': (
                                'eNpljwEKACAIA/cy19N6WX4tSzQrQczbhgSs6mo9gEabxGFiTJ3tVt8/RveuPfVRfP'
                                'GWh/HOpi6uB4t8ZkMvN9rjrXf3f3j7DpsDRzZw'
                            ),
                        },
                    },
                    '3dd6cbfb2425dc5ec2c2f10ab4f7865c6f06316ac9ccfa40fa21fd2be8d1d05d': {
                        'tests': ['test_node_list.py::TestPieces::test_cleaning_is_rns[3]'],
                        'inputs': ['<f4[17, 5]', '<f4[3, 2]', '<f4[1, 2]'],
                        'value': {
                            'array': '<f4',
                            'shape': [15, 5],
                            'zlib64': (
                                'eNp1zsENACAIA0Amo4zGZLiaUUHBqEk/FwoSaaMZ8Shom9rDePmw8e65MMHH4PvwMD'
                                'v/2Bb32IPUSXejN28hGapFp5jvLl3ccx3P+zsx'
                            ),
                        },
                    },
                    'a0befeb3a4114eb14f3632ffab6a2fa92b609154af95e57475a1eda6248e6a13': {
                        'tests': ['test_node_list.py::TestPieces::test_cleaning_is_rns[6]'],
                        'inputs': ['<f4[31, 5]', '<f4[3, 2]', '<f4[1, 2]'],
                        'value': {
                            'array': '<f4',
                            'shape': [19, 5],
                            'zlib64': (
                                'eNptj4ENACEIA5kMGM3JYDUR0RT/TUjM0RYgGk67OMqihC7T+Ktvtl4y/7L02aMD78'
                                'lcLOcI5ODcylHpLLMwj0EL7OiO93qqN7BvD+Oe15jALl43St+5Mf5nKsgmuQtNzA=='
                            ),
                        },
                    },
                    'a1d9ab854b1ba019681a0f5e1b0f9a012582b6c48c703eea38e29b600ce36531': {
                        'tests': [
                            'test_node_list.py::TestPieces::test_cleaning_reaches_its_fixpoint',
                        ],
                        'inputs': ['<f4[61, 5]', '<f4[3, 2]', '<f4[1, 2]'],
                        'value': {
                            'array': '<f4',
                            'shape': [41, 5],
                            'zlib64': (
                                'eNpNklsNxDAQxAJhoQyBPiAESiEESojclUqg3KXdUfyxGstq+2G1lOsu5dzLs/2/7V'
                                'g8d8CNdHEsF7mCU26Fq7kX3OR6v9zANc/OJzilC7hIN77LTZ7b4Xq6E+5M1z7LTX52'
                                'g9vy+R3v7vlduJ5uwI107meu6Geu6Geu6GcW+pmFfkI7wfkCLtK5n1noZxb6mYV+Zq'
                                'GfWehnFvqZhX5moZ9Z6GcW+pmFfuZAP3OgX+A/e/v9ADWnuTc='
                            ),
                        },
                    },
                    'a7bd8898eeca75d75e7ee3fafca76189841264529c159ed9045c6cb829a46464': {
                        'tests': ['test_node_list.py::TestPieces::test_cleaning_is_rns[2]'],
                        'inputs': ['<f4[22, 5]', '<f4[3, 2]', '<f4[1, 2]'],
                        'value': {
                            'array': '<f4',
                            'shape': [10, 5],
                            'zlib64': (
                                'eNpjYGjYzwDGDiBsD2QDMQg0wPj7IGIgeRAbTNshidlB8T5UMXS9DkhmgMRgelDMs8'
                                'cthmIHVAzsTrCbAQsNKr8='
                            ),
                        },
                    },
                    'e9740ef1621dea9a858a9bcb98260b214701824562a8c80a284e4e361d661731': {
                        'tests': ['test_node_list.py::TestPieces::test_cleaning_is_rns[0]'],
                        'inputs': ['<f4[32, 5]', '<f4[3, 2]', '<f4[1, 2]'],
                        'value': {
                            'array': '<f4',
                            'shape': [19, 5],
                            'zlib64': (
                                'eNplkIENwCAIBJkMGI3J6mpWi3pUk0/kzPOgSDSZ8iF77yabhd5s1k8yg7d9tbcqMv'
                                'ZzZiqYHcV608zUyuglo9fZw8783GNn4g/KHCsjZxgnfhlkq/fFircDKndHzg=='
                            ),
                        },
                    },
                },
            },
        },
        'agree': True,
    },
    'node_list.rn_forward': {
        'platforms': {
            'arm64': {
                'research': '7440581b174be13887db6c391e45efc9c3ee2668',
                'recorded': '2026-10-22',
                'jax': '0.6.1',
                'jaxlib': '0.6.1',
                'numpy': '1.26.4',
                'xla_flags': ['', '--xla_cpu_multi_thread_eigen=false'],
                'values': {
                    '19c77286a3d730a8bbc8582771adf5f06b6f98874ec8791bc576882ea18f5070': {
                        'tests': ['test_node_list.py::TestPieces::test_forward_is_rns_rule'],
                        'inputs': ['<f4[8, 1]', '<f4[20, 3]', '|b1[20]', '<f4[4, 3]', 3, 1, 9],
                        'value': {
                            'array': '<f4',
                            'shape': [4, 1],
                            'zlib64': 'eNrzNoq19wZiP6tIMAYAJagEcw==',
                        },
                    },
                },
            },
            'x86_64': {
                'research': '7440581b1',
                'recorded': '2026-10-23',
                'jax': '0.6.1',
                'jaxlib': '0.6.1',
                'numpy': '1.26.4',
                'xla_flags': ['', '--xla_cpu_multi_thread_eigen=false'],
                'values': {
                    '19c77286a3d730a8bbc8582771adf5f06b6f98874ec8791bc576882ea18f5070': {
                        'tests': ['test_node_list.py::TestPieces::test_forward_is_rns_rule'],
                        'inputs': ['<f4[8, 1]', '<f4[20, 3]', '|b1[20]', '<f4[4, 3]', 3, 1, 9],
                        'value': {
                            'array': '<f4',
                            'shape': [4, 1],
                            'zlib64': 'eNrzNoq19wZiP6tIMAYAJagEcw==',
                        },
                    },
                },
            },
        },
        'agree': True,
    },
    'node_list.rn_run': {
        'platforms': {
            'arm64': {
                'research': '7440581b174be13887db6c391e45efc9c3ee2668',
                'recorded': '2026-10-22',
                'jax': '0.6.1',
                'jaxlib': '0.6.1',
                'numpy': '1.26.4',
                'xla_flags': ['', '--xla_cpu_multi_thread_eigen=false'],
                'values': {
                    '12b63233be3e65e553906cbcd914dcddfe412f8d5bb5944fe22970dfc9d081e3': {
                        'tests': [
                            (
                                'test_node_list.py::TestAgainstLiveRN::'
                                'test_rn_populations[2-with_self-2-None]'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'max_depth': 2,
                                            'band_threshold': 0.3,
                                            'max_weight': 8.0,
                                            'recurrence': {
                                                'enabled': True,
                                                'allow_hidden_to_hidden': True,
                                                'iteration_level': 2,
                                                'allow_backward': False,
                                                'allow_lateral': False,
                                                'allow_self_loops': True,
                                            },
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            2,
                            'rn',
                            1e-06,
                        ],
                        'value': {
                            'dict': [
                                [
                                    'connections',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNrrY2BgiATiWCCuAmIhIHYE4gIgLgNiLiDWAGI9IBYAYmEgjg'
                                                'ZibyDmAOIgIJYE4nggdgJiAPZYBTU='
                                            ),
                                        },
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNoNwyEOgDAQALAThEwuoPaDfWOzGIJigqDwKN5Pm/SLiGq2uP'
                                                'iavBweNldvu5uTp7uPsz/CJASx'
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'fitness',
                                    [
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjYGCwD2ZksJ8DxAxgoGDPAxTbGstgX/OzHiQGxAb2z8Re2T'
                                                '34r2/PCeZDMIcygz0/mK1gzwEVAwBAWg1f'
                                            ),
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNqr+Vlvz8BgYP/gvz6QVgDjrbEM9hzKDPZzGBnsg4GYn4HBng'
                                                'eIOYGYA4gZoLhMyh5MczE9sOv5X28H0gsAsf8PnA=='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'graphs',
                                    [
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//RdPHodm1yw5vbMLvd3eK4rPX/7TT/kFZMbPIF5Gl5'
                                                'ZI+M1fjR4uNuKHTIG0QKItvqnUDKwdwKgwSKmpV0/XEO1n6ueu'
                                                'YXcMxR4kMbnG6D8swWEEDMd90FHTQ4u8ifK/t+FIYaLN3e8zLQ'
                                                'ENRr00I+4UPipJMbLMM4RfMT0Ym7HUa3XdRJQmDyFlO/zYpgzt'
                                                'SSBJT5QVbIu0ceCIN+18Ie6BUuxMcvD9j4f8B4nwnV3SBWTkCp'
                                                'g69VOF3aXyAgmzxYQ4pw3oJufq2/7IY4WZ15WgReSMVVFmhUVh'
                                                'f+erT61GHs8erEgh8diJAG1qGzlpJIuKm7RQ1m1s88Z1Utyn2+'
                                                'JB4yuRjPcH4tmEsZdZmQ4VAmkcF5iEM+JUl4cRCUjrjpdZhZQC'
                                                'ItPvyOpS3Q9oblsFt/pcYLtxqBVx/qRpfD56ye+ayv8nvPyFCn'
                                                'Gh+sxgA9c2BBuEvJun7Sod1+6GSJ91pLUrbZXj283aZ3kzG5Us'
                                                'VJq5j8/05C/zCs/SZIkVy//ntpoziMg70acoF7AcFJ5rkA8p7f'
                                                'NCCVZl2+h5xgb/gpbq7DnfjxOcyMfi4syFGV0tHZ+LlXYTmDQW'
                                                'tXuMkmEXG49BH4+UDWBQsYGs7tpfUF3MRccHj+JsiGm4LiR0MN'
                                                'daDcMl9OZjEvgy6rTuxcY4ZLKYHHExV/5c8eKon64MVmO8RSYn'
                                                'ykQyQSfqn0zcNGIJMHAmHxChmRjn8D3YPbNqFWdFU5QPjaqAh7'
                                                'E3tHteTLbVBglTaJ1s6NPRz3nvfCPutEbVLCKxGfMPVYENA5+4'
                                                'JMHJUsPjMGJuISdf0dXfiybaahVjeFQbxOM5rJclkuvwH1XdjB'
                                                'szQELjIG90eJhJlj3u'
                                            ),
                                        },
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//R7PHqxIIfHYiQBtahs5aSSLipu0UNZtbPPGdVLcp9'
                                                'vil1mFlAIi0+/I6lLdD2huWwW3+lxgu3GoFXH+pGl8Pnp1pLUr'
                                                'bZXj283aZ3kzG5UsVJq5j8/05C/zCs/SZIkVy2oVZ0VTlA+Nqo'
                                                'CHsTe0e15MttUGCVNonWzo09HPee98a3XdRJQmDyFlO/zYpgzt'
                                                'SSBJT5QVbIu0ceCIN+18Ie6EOKcN6Cbn6tv+yGOFmdeVoEXkjF'
                                                'VRZoVFYX/nq0+tRtdaDcMl9OZjEvgy6rTuxcY4ZLKYHHExV/5c'
                                                '8eKon64MZ+rnrmF3DMUeJDG5xug/LMFhBAzHfdBR00OLvInyv7'
                                                'f4zV+NHi424odMgbRAoi2+qdQMrB3AqDBIqalXT9cQ7VZjvEUm'
                                                'J8pEMkEn6p9M3DRiCTBwJh8QoZkY5/A92D2zgVLsTHLw/Y+H/A'
                                                'eJ8J1d0gVk5AqYOvVThd2l8gIJs8X/57aaM4jIO9GnKBewHBSe'
                                                'a5APKe3zQglWZdvoecYG/yPutEbVLCKxGfMPVYENA5+4JMHJUs'
                                                'PjMGJuISdf0dXfiybaahVjeFQbxOM5rJclkuvwH1XdjBszQELj'
                                                'IG90eJgXG49BH4+UDWBQsYGs7tpfUF3MRccHj+JsiGm4LiR0MN'
                                                'gPE6BfGUWdYssnG53Wgb/SDoiYlT6Q9Ra4BQTByYboRETtzxYK'
                                                'O5E0KosYmmBTnpnOHskiuF2sm9X3QYA9XWncWjBZhuhLj0igwW'
                                                'j40IZJ9MCnJ25dH+5WreyLZjzdS+KEbolMvCfKmneFLC29yXnw'
                                                'Vq8AGfCZEw/PR5VOif0ziy0k0Teu4U2ngr+1QaaQPgk0wf4aXh'
                                                'E8pkbyPamFu9sZsD5Y'
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'invented',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYMAEAAAUAAE=',
                                        },
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYEABjCACAAAZAAI=',
                                        },
                                    ],
                                ],
                                [
                                    'populations',
                                    [
                                        (
                                            '6fba85ebaa5734e567c55e9b2574cc795d717e9939d24d6ceafb5b'
                                            '2d8450db80'
                                        ),
                                        (
                                            'f8261f58da82c8fbc770366d89fae846c3488be512ebf5d47c1644'
                                            'c35906ae14'
                                        ),
                                    ],
                                ],
                                [
                                    'reported',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNrrY2BgiATiWCCuAmIhIHYE4gIgLgNiLiDWAGI9IBYAYmEgjg'
                                                'ZibyDmAOIgIJYE4nggdgJiAPZYBTU='
                                            ),
                                        },
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNoNwyEOgDAQALAThEwuoPaDfWOzGIJigqDwKN5Pm/SLiGq2uP'
                                                'iavBweNldvu5uTp7uPsz/CJASx'
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'stable',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjZMQEAADmABU=',
                                        },
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjZMQEAADmABU=',
                                        },
                                    ],
                                ],
                            ],
                        },
                    },
                    '1e186bb449624fe1f58832593a0149d0b567d60a5752dcab2e97c5283a845e43': {
                        'tests': [
                            (
                                'test_node_list.py::TestAgainstLiveRN::'
                                'test_depth_three[hidden_only-None]'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'max_depth': 3,
                                            'band_threshold': 0.3,
                                            'max_weight': 8.0,
                                            'recurrence': {
                                                'enabled': True,
                                                'allow_hidden_to_hidden': True,
                                                'iteration_level': 2,
                                                'allow_backward': False,
                                                'allow_lateral': False,
                                                'allow_self_loops': False,
                                            },
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            2,
                            'rn',
                            1e-06,
                        ],
                        'value': {
                            'dict': [
                                [
                                    'connections',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNp7zMrA8JKJgWEDMwPDUhYGBjEGBoYXQL4pUJwDiKOBfAUg/g'
                                                '3E7kA8CYgbgPIzgdgRyJ4ApEFAH6j3GiMDAwDDUgnA'
                                            ),
                                        },
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjYGBgaGBiYNjAzMDgDmQrALEYEJuyMjC8AIpfY2RgmACkHY'
                                                'FiM4H0JCAdDcT6LAwMi4D8UKA8kGIQAtJALQwASr4HUA=='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'fitness',
                                    [
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNoT+NNsF7P3mB0Dg4O93/L/QFrBXoaBwT6amcG+yWwKkM9gzw'
                                                'gUY2F4AGQb2IP4ILUQmgHO33TiP1gtAAYgEJ0='
                                            ),
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjYHCwZ4BjA3tGBgUgrWAfzcxgL8PAAGRjx5tO/LeDsA3sZ4'
                                                'nYgcVCnRjsAXVnCsM='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'graphs',
                                    [
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//WYXl7QPL9JHb6RdkA1oQRJECPfBiICDNe7JfOEU6E'
                                                'vZMLH/r8UqhM+cbx3Hx/h5eFT3GyMml9O/YqkauzoW/zXEj2+F'
                                                'Fv/jOXS/pemFE6HeRDbO9kA4GkTkpbFCNujZ9FEJ0uJzJrYT44'
                                                'i6Ph2zPfP/JXXxNebTkSJcjSOdGGDkwy9/kEdlkKVmgqvyMfmi'
                                                'YGcrGOIBYe2zqdcGLZ4ay6vD9w8ZRCnB1o6iNeLwf3stIKGPP2'
                                                'rzTWcy09y4dFBFH7AKQqz5m+LeCbWC5prDUNR0NxZeLpV+f5AT'
                                                'WzfGwGLRUgCLdbIIPY+NkrQSlg2QelxJVQumi7/mhy06iA2WJH'
                                                '50Td+2qqEoTUr5VZlbOi3H1UPMUwEkFGW7jRNTfdSQKhv+DheH'
                                                'tSDPywGzssbb/l9PtcynyZyG+r/v/jJ6FaJAjR/0WBdkdxbSGh'
                                                'LWkvpuxIr2Wc6pTPQf6ZUE5txVcVvC5o66i//nqcBKKmoJGFXt'
                                                'F81gDE+J7jibJm/+nXqc28c//8NIxQJG2Wqx0QEkdszpqTLA0e'
                                                'JhaLI71F+s1hDMR17Cv43iTXwcAR9/dOZmL4vnT9qimPV0aCuD'
                                                'OqiL2R/sgVvQu+ZECDpwTFqb9YmbWFtorPvqQHfdu9D6I1g9+c'
                                                'YEhG9w7Ymwb12KY8a+Mn7ZnztZY1/7o/AllrH6H/L/SNtB/won'
                                                'L3E1XzkhJ6BfvPA3PvDJxN3vPlhjOijllmvjsMRCmPwcFJr79M'
                                                'iZb7kkJ65B5GSbk0yklZkbeFK4VTKJE7W064P6BvFlHaPmSYNC'
                                                'iUqNb6dsn5GRfN4keAuzkNIib9SDtW12e8zUKc0YGYKKDfxAU2'
                                                'MYqF0Dp2xh8MRFHkYU'
                                            ),
                                        },
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//eOwxEKY/BwUmvv0yJlvuSQnrkHkZJuTTKSVmRt4Ur'
                                                'hVR17Cv43iTXwcAR9/dOZmL4vnT9qimPV0aCuDOqiL2R/Ej2+F'
                                                'Fv/jOXS/pemFE6HeRDbO9kA4GkTkpbFCNujZ9MLmjrqL/+epwE'
                                                'oqagkYVe0XzWAMT4nuOJsmb/6depzbG/4OF4e1IM/LAbOyxtv+'
                                                'X0+1zKfJnIb6v+/+MnoVokDDL3+QR2WQpWaCq/Ix+aJgZysY4g'
                                                'Fh7bOp1wYtnhrLq7AKQqz5m+LeCbWC5prDUNR0NxZeLpV+f5AT'
                                                'WzfGwGLRw/cPGUQpwdaOojXi8H97LSChjz9q801nMtPcuHRQRR'
                                                '+Q0iJv1IO1bXZ7zNQpzRgZgooN/EBTYxioXQOnbGHwxEjbQf8K'
                                                'Jy9xNV85ISegX7zwNz7wycTd7z5YYzoo5ZZrxgSEb3DtibBvXY'
                                                'pjxr4yftmfO1ljX/uj8CWWsfof8v/sgVvQu+ZECDpwTFqb9Ymb'
                                                'WFtorPvqQHfdu9D6I1g9+cc//8NIxQJG2Wqx0QEkdszpqTLA0e'
                                                'JhaLI71F+s1hDMdE3ftqqhKE1K+VWZWzotx9VDzFMBJBRlu40T'
                                                'U33UkCoyiRO1tOuD+gbxZR2j5kmDQolKjW+nbJ+RkXzeJHgLs/'
                                                'eOjlNtgckHSFsuJDAxURod0G4HxwIm2VvZizTzv1TIo96mJtqr'
                                                'XDreLFFMaBeaxeSq2PLCrxK9iaZq2+TGa0GtTEfkOMss2EXP2A'
                                                '7FIRQWMpvIbrSvVDBuntQ8Ybu/zZuz9Rb9CElTdR385V5c8yt8'
                                                'rUEYOgI/IHFtpVv4DEPsEqt74TV4T6ZPBP39/WoGPnAo5hCXRd'
                                                'gbatamO/2sWX73M0cG'
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'invented',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYEACjBAKAAAbAAI=',
                                        },
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYGRABYyMjAAALQAF',
                                        },
                                    ],
                                ],
                                [
                                    'populations',
                                    [
                                        (
                                            '6fba85ebaa5734e567c55e9b2574cc795d717e9939d24d6ceafb5b'
                                            '2d8450db80'
                                        ),
                                        (
                                            'f29ad957c49dbe2aa0b61796a88dc2ac1a87255a9a0eb7a433c443'
                                            'd9ecc47cb9'
                                        ),
                                    ],
                                ],
                                [
                                    'reported',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNp7zMrA8JKJgWEDMwPDUhYGBjEGBoYXQL4pUJwDiKOBfAUg/g'
                                                '3E7kA8CYgbgPIzgdgRyJ4ApEFAH6j3GiMDAwDDUgnA'
                                            ),
                                        },
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjYGBgaGBiYNjAzMDgDmQrALEYEJuyMjC8AIpfY2RgmACkHY'
                                                'FiM4H0JCAdDcT6LAwMi4D8UKA8kGIQAtJALQwASr4HUA=='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'stable',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYGBgYARhEAAxgRAAAF8ACg==',
                                        },
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjZGBgZGQEEiDICOYwAgAAhwAN',
                                        },
                                    ],
                                ],
                            ],
                        },
                    },
                    '20bb6f8cccd0354c9da0002ab1e72942f9f5217e799ee587e50626d5da16c446': {
                        'tests': [
                            (
                                'test_node_list.py::TestAgainstLiveRN::'
                                'test_rn_populations[1-with_lateral-2-None]'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'max_depth': 1,
                                            'band_threshold': 0.3,
                                            'max_weight': 8.0,
                                            'recurrence': {
                                                'enabled': True,
                                                'allow_hidden_to_hidden': True,
                                                'iteration_level': 2,
                                                'allow_backward': False,
                                                'allow_lateral': True,
                                                'allow_self_loops': False,
                                            },
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            2,
                            'rn',
                            1e-06,
                        ],
                        'value': {
                            'dict': [
                                [
                                    'connections',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNotyUsKACAMA9GgotiFP+j9r+oUungwJC7pYuGhZcd2MNBRk/'
                                                'KLLnBMbBg+J8gA2g=='
                                            ),
                                        },
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNoTZIAAZihmA2JBKFsUiFmAmBOIBaB8PiBmAmIuIBaB6gXJC0'
                                                'FpAB8EAMs='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'fitness',
                                    [
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNq7bKhgz8zgYM/AwGBvEN5ox8SgYH9Ni8G+IUfJvmYhu70AUH'
                                                'wZr7r9kxAT+w/aBkB1DvZcQDG3pQr2V2QZ7JWWMgHFFOw53O7Y'
                                                'bWZmsAcABT4TXg=='
                                            ),
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjZnCwZwDiJyEm9h+0DeyX8arbN+Qo2bstVbC/bKhgz8SgAJ'
                                                'RXsK9ZyG6vtJTJ/poWg/0VWQb7zcwM9nP+2YP1mmra2DvlO9hX'
                                                'JSrYAwBkBhWz'
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'graphs',
                                    [
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//dRDxF0txIZUAIsNKRzbDoj4FIA2CxJR18e6iqBX5x'
                                                'QwijM31PqYC693a1GLKjpqeAbgdNQ7dMKxJM24Bw2XL81rfBlG'
                                                'LbFg5L2FMI+v/sY90rmkPqkUbo+2KK5q4Kfp5VGYgx9lwnAUjG'
                                                'J/0uwslrgKZ2+EnT0XHBKSDzFvmCtRLcicdY3XbeXzzdLmem3q'
                                                'pcngv92mnQ9l15Qj00ZDC/l/20PCb3Dll9EbUMUd9BRamqor8e'
                                                'TpARVzKEroi/ffZUTPlEyNG49QG6ehcrY2uY/FfIv9idosBQuk'
                                                '1bgjLH3tT1m4+cLth8lY6ndMUn310uJbAi3T+afxwid1srs/+2'
                                                'n1UcNQ7I6sPtEQGOWu8txFhRC72hSoVRBFXRiBseBn+A1xOC38'
                                                '8xMnXeZ/eHrTke/EtTOnKX1UDVumGVSUsCiPetf8bDF6LiSCje'
                                                'jdXkoYIPvuBXhtZSr74KoTHRAa0yZVTSZlBJ8aCBG25h/Dg/d2'
                                                'GqQHwe1YPSYzKVL+/52weuOwxEKY/BwUmvv0yJlvuSQnrkHkZJ'
                                                'uTTKSVmRt4UrhVPTNePO5Qi5UNuDAI00ubj9T80Yl9s9mWPJmi'
                                                'iva/wBjE1d/n/oceGDZFcIjoSzMKzSBO7uGTb3nrYyL+0ryj7I'
                                                'SbyQOLuc1Hbj6pte82LLrY7kwh9m7zo4TrGLbd3I/EH7hEYufz'
                                                '7q1zfRRnKWdEqBWRPXua7HLNuwyZMGlmKsui7t0Zv5vnNlVMJ5'
                                                'f18Ys/90j+bay/SbcswyqjmvdJDFcr8AOEBt+KdqH8BB6SIddC'
                                                '/Q1FRS6N9E3VQ0y9qSv4ltjKJ8BXH2nRxcgeAYVSllbvz3aScu'
                                                '7ZfOtvDZQR3t/TuEcb'
                                            ),
                                        },
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//YozN9T6mAuvd2tRiyo6angG4HTUO3TCsSTNuAcNly'
                                                '/N47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFV61/xs'
                                                'MXouJIKN6N1eShgg++4FeG1lKvvgqhMdEBrTJlVNJmUEnxoIEb'
                                                'bmH8OD93YapAfB7Vg9JjMpUv7/nbB6DXE4LfzzEydd5n94etOR'
                                                '78S1M6cpfVQNW6YZVJSwKI9Ez5RMjRuPUBunoXK2NrmPxXyL/Y'
                                                'naLAULpNW4Iyx97cTV3+f+hx4YNkVwiOhLMwrNIE7u4ZNveetj'
                                                'Iv7SvKPs1EPEXS3EhlQAiw0pHNsOiPgUgDYLElHXx7qKoFfnFD'
                                                'AtyJx1jddt5fPN0uZ6beqlyeC/3aadD2XXlCPTRkML+aLu3Rm/'
                                                'm+c2VUwnl/Xxiz/3SP5trL9JtyzDKqOa90kMT1m4+cLth8lY6n'
                                                'dMUn310uJbAi3T+afxwid1srs/+2kfuERi5/PurXN9FGcpZ0So'
                                                'FZE9e5rscs27DJkwaWYqy3/bQ8JvcOWX0RtQxR30FFqaqivx5O'
                                                'kBFXMoSuiL999lhJvJA4u5zUduPqm17zYsutjuTCH2bvOjhOsY'
                                                'tt3cj8SW2MonwFcfadHFyB4BhVKWVu/PdpJy7tl8628NlBHe31'
                                                '201zNkgRYZiU1Pfa+Zo0EcNqcpui4QNx4k7IvvxjcP47DEQpj8'
                                                'HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFUDqebNpETGhy2R04'
                                                'eRl6FqSq8IWgrQSCwnPDv9jPxJ4MIzvRIRrQ3vu8bQ7CtLDwxd'
                                                '3bUw09PjoLdp1Ews1peAPYI1LpfNBaR8bw0cGavm2sxiB0aJmO'
                                                'EHNptpnbg2KjclfUSE'
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'invented',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYIAARgYEAAAAIAAC',
                                        },
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYEAFjIwMDAAAGwAD',
                                        },
                                    ],
                                ],
                                [
                                    'populations',
                                    [
                                        (
                                            '6fba85ebaa5734e567c55e9b2574cc795d717e9939d24d6ceafb5b'
                                            '2d8450db80'
                                        ),
                                        (
                                            'cb1cfa38706576290397451c3bf9a78b030cb2fc896c8bbc850896'
                                            '4503ab5847'
                                        ),
                                    ],
                                ],
                                [
                                    'reported',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNotyUsKACAMA9GgotiFP+j9r+oUungwJC7pYuGhZcd2MNBRk/'
                                                'KLLnBMbBg+J8gA2g=='
                                            ),
                                        },
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNoTZIAAZihmA2JBKFsUiFmAmBOIBaB8PiBmAmIuIBaB6gXJC0'
                                                'FpAB8EAMs='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'stable',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjZMQEAADmABU=',
                                        },
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjZMQEAADmABU=',
                                        },
                                    ],
                                ],
                            ],
                        },
                    },
                    '245c81906facbd5e4cc86dbc3caed04998cd7cdb5a82ab9f53a5667557062d08': {
                        'tests': [
                            (
                                'test_node_list.py::TestAgainstLiveRN::'
                                'test_rn_populations[2-full_recurrent-3-20]'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'max_depth': 2,
                                            'band_threshold': 0.3,
                                            'max_weight': 8.0,
                                            'recurrence': {
                                                'enabled': True,
                                                'allow_hidden_to_hidden': True,
                                                'iteration_level': 3,
                                                'allow_backward': True,
                                                'allow_lateral': True,
                                                'allow_self_loops': True,
                                                'activate_time': 20,
                                            },
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            2,
                            'rn',
                            1e-06,
                        ],
                        'value': {
                            'dict': [
                                [
                                    'connections',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNp7x8DAcASI5wLxeiCWAOKHQPwdiO8AsTEQOwFxJRDLAPEcIL'
                                                '4J1eMHxB+A2B2q9hoQAwAJFgxF'
                                            ),
                                        },
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNp7x8DAMAeIZYC4EoidgNgdiCWA+DsQH4Hih0D8AYivAfEdIP'
                                                'YD4pNAfBqILwHxEyBOAWIA7gcMvw=='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'fitness',
                                    [
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjYHCwn9rDYC/D2mDHwMBgz8CgYM8DpM8EC9tD+CBsAMc99X'
                                                'ZgsRPCTGCaDaoHphYA9d4LKA=='
                                            ),
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjYHCw76m3s2dgMEDCCmB8JljY/oQwk/3UHgZ7HgYGezYgZk'
                                                'DCt2Qb7UDqWYHsq1GMYD0AKwMNhA=='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'graphs',
                                    [
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//ZzoaqUt2GT2rsCA1qaUsyQvnBcd1/0bj5VJyPxwGL'
                                                'JH41tQqzVp6i8h6kPsswX2GdEboSVupftV3rVwWhSVCOcN/AiV'
                                                'PdL+RdjQb9JFDwVrnA4slHNcGPEsff+Vv3spOe55BXXeLnfhqe'
                                                'lJ/Wnbgt/uTBeH4OZmTYLTGdZZrwPEex+/suS5VgKYRH3UCWp/'
                                                'ORiefT0qgeagP5nq2IXJx0ji4blpPXjdPveF7Qf39HCLksmZIH'
                                                'wmbXSclLe6mApIy4uihfJaJ/uHPaNuioGdsFTgf4pOXvZkSkyS'
                                                '1470gFgBZ46HeFpp9wvu5oHaTHbUmPW4j9DPrsMfoVglQW6ZrQ'
                                                'JU7zD0FGYqXMTMCfbObUvNztahDgw/Qg59xQhjCjzWM5aLhpsV'
                                                'Ax11HAwLwED/K58tGnFzsoP8H51otqbhPDlwSN5McfkSjG3hr2'
                                                'hEYr1jqbXPKX/FXKlfioP9OT4UV5iSfzYfQt7LFwotKYbxXBUn'
                                                'h6w9k6i9BtSdSA1M02ISfx4dc8YIj1nmoL9EUbWkyAOIOm7EFU'
                                                'DNhahMiDh3iuCeSFi/3HJ2LczGER8pDVsOLyWeAJBcQa13Z2sF'
                                                'piM7OG38EpUKkjhwF+Sm+zsh1pWS/GEph5vCQXtLPx8v0I7/ge'
                                                't6vteKvmlyEBQfu+SY07IwZ537M7im/wcnRENQSeTcrOi7xyes'
                                                'Ci0DLckN79PgpTaIz1WWUKY2jx77jBWHr/jQsvuUI6C712V9Pa'
                                                'pPQ6CvviwyiOpGRlrz45XTFpgG9wAih6w/9eMX93rw5v3H3k6v'
                                                'H5RkJelx9MhSB+eKasytjp/JRZ5OoyR+SQaxJoTqDX68/Z6l67'
                                                'Gs9KYQwahu52LcYUD8'
                                            ),
                                        },
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//ZzoaqUt2GT2rsCA1qaUsyQvnBcd1/0bj5VJyPxwGL'
                                                'JHHh1zxgiPWeagv0RRtaTIA4g6bsQVQM2FqEyIOHeK4J6SfzYf'
                                                'Qt7LFwotKYbxXBUnh6w9k6i9BtSdSA1M02ISf0jeTHH5Eoxt4a'
                                                '9oRGK9Y6m1zyl/xVypX4qD/Tk+FFeYlouGmxUDHXUcDAvAQP8r'
                                                'ny0acXOyg/wfnWi2puE8OXDQsvuUI6C712V9PapPQ6CvviwyiO'
                                                'pGRlrz45XTFpgG93sfv7LkuVYCmER91AlqfzkYnn09KoHmoD+Z'
                                                '6tiFycdIi6KF8lon+4c9o26KgZ2wVOB/ik5e9mRKTJLXjvSAWA'
                                                'H8EpUKkjhwF+Sm+zsh1pWS/GEph5vCQXtLPx8v0I7/geNbUKs1'
                                                'aeovIepD7LMF9hnRG6ElbqX7Vd61cFoUlQjn4uG5aT143T73he'
                                                '0H9/Rwi5LJmSB8Jm10nJS3upgKSMus6LvHJ6wKLQMtyQ3v0+Cl'
                                                'NojPVZZQpjaPHvuMFYev+I6fyUWeTqMkfkkGsSaE6g1+vP2epe'
                                                'uxrPSmEMGobudiACKHrD/14xf3evDm/cfeTq8flGQl6XH0yFIH'
                                                '54pqzK3rer7Xir5pchAUH7vkmNOyMGed+zO4pv8HJ0RDUEnk3F'
                                                '82yAI0y5OjPSEooWORsV5tGt/EB0VRm1qPImUTEafpCWfWidi/'
                                                'AiqOxuNfq0IbDccM6FM5PODCBvuePJNtEcfCYSItIaVrjwzAtu'
                                                'u7kl/IhliK37kKOLw41WeTehPywS++TOklrIEeezFYmubCkPM6'
                                                'Ern/Qp9pNy/MWLG0cLMc2IVCAmPJIhhEeuSiNIGFNeA+mtNlbi'
                                                'SllFnkoyIEekrlgT+g'
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'invented',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYMAEAAAUAAE=',
                                        },
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYMAEAAAUAAE=',
                                        },
                                    ],
                                ],
                                [
                                    'populations',
                                    [
                                        (
                                            '6fba85ebaa5734e567c55e9b2574cc795d717e9939d24d6ceafb5b'
                                            '2d8450db80'
                                        ),
                                        (
                                            'c506c5c74656b565ec9eb88362a95205d34b37774a5c04d2b78a6f'
                                            '88866abeb3'
                                        ),
                                    ],
                                ],
                                [
                                    'reported',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNp7x8DAcASI5wLxeiCWAOKHQPwdiO8AsTEQOwFxJRDLAPEcIL'
                                                '4J1eMHxB+A2B2q9hoQAwAJFgxF'
                                            ),
                                        },
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNp7x8DAMAeIZYC4EoidgNgdiCWA+DsQH4Hih0D8AYivAfEdIP'
                                                'YD4pNAfBqILwHxEyBOAWIA7gcMvw=='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'stable',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjZMQEAADmABU=',
                                        },
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjZMQEAADmABU=',
                                        },
                                    ],
                                ],
                            ],
                        },
                    },
                    '3c6dbb685eda5a21d551d8416a8c97a9c0732b4492cb4bca5ba00cb7352f69df': {
                        'tests': [
                            (
                                'test_node_list.py::TestAgainstLiveRN::'
                                'test_invented_connections_are_not_ported'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'max_depth': 2,
                                            'band_threshold': 3.0,
                                            'max_weight': 8.0,
                                            'recurrence': {
                                                'enabled': True,
                                                'allow_hidden_to_hidden': True,
                                                'iteration_level': 2,
                                                'allow_backward': True,
                                                'allow_lateral': True,
                                                'allow_self_loops': True,
                                                'activate_time': 20,
                                            },
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            1,
                            'rn',
                            1e-06,
                        ],
                        'value': {
                            'dict': [
                                [
                                    'connections',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': 'eNpjYYAAXSjNgoRdkMRgtD4SH1kuCEoDACk4ARM=',
                                        },
                                    ],
                                ],
                                [
                                    'fitness',
                                    [
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNork7K3Z2BwsD+jqQCmy4B8GJZqTYeLweiWO95wPrLcxrvSYB'
                                                'oA+KkS1A=='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'graphs',
                                    [
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNq7wS+8IF7SdW7SaXXpudca91/i65gx1W7CV7EdrCwHT7a9eL'
                                                'zhiNOMPzIis35/OTEzf6eK+jrHJymzJ/ssmTpTuiJoR2hzppHe'
                                                'pqdik4y6jCeI88Wo7OwSNbyVZpZTkhrecSFo3itC+m8QsJ9S+a'
                                                'snHr904Cr89aw4SXaaZnWUn8vsPY8er36rsm961ptX65IodR+l'
                                                '+s/H758op6m9qfir9CERhjVnxHyXVZeu/XBi3yq9ePeNBsWU2k'
                                                '9r/63WWvQ5cGJehxzbjjO90c58IqxGm3SWrJJa+8D6h++iVTmE'
                                                '9AMA00NBXw=='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'invented',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjZGRgBAIwAQEMjAAAxQAS',
                                        },
                                    ],
                                ],
                                [
                                    'populations',
                                    [
                                        (
                                            '6fba85ebaa5734e567c55e9b2574cc795d717e9939d24d6ceafb5b'
                                            '2d8450db80'
                                        ),
                                    ],
                                ],
                                [
                                    'reported',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': 'eNpjYYAAXSjNgoRdkMRgtD4SH1kuCEoDACk4ARM=',
                                        },
                                    ],
                                ],
                                [
                                    'stable',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjZAQDBkYkAAAA2QAU',
                                        },
                                    ],
                                ],
                            ],
                        },
                    },
                    '45059ad48d060caff480313533b6b382e27594d8e570e179d65f029dfad9a412': {
                        'tests': [
                            (
                                'test_node_list.py::TestAgainstLiveRN::'
                                'test_rn_populations[2-hidden_only-2-None]'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'max_depth': 2,
                                            'band_threshold': 0.3,
                                            'max_weight': 8.0,
                                            'recurrence': {
                                                'enabled': True,
                                                'allow_hidden_to_hidden': True,
                                                'iteration_level': 2,
                                                'allow_backward': False,
                                                'allow_lateral': False,
                                                'allow_self_loops': False,
                                            },
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            2,
                            'rn',
                            1e-06,
                        ],
                        'value': {
                            'dict': [
                                [
                                    'connections',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNprYGBg8AFifyDOBWI+ILYA4gQonwOIFYFYHYh5gVgAqtYJiN'
                                                'mA2BOIxYA4FIitgRgA12AEjw=='
                                            ),
                                        },
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpTZGBgyAViPiDmBWIxIFYH4gQgZgNifyD2AWILIPYEYgEgtg'
                                                'ZiJyDmZoAAIyCug7IBm+gDsQ=='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'fitness',
                                    [
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjYGCwD2FksJ8DxAxAdruBlT0PkHZNAvEdwGLHFfLt34Qz2v'
                                                'NLqtpzMkDUgTCHMgOUr2Cf1PfNDiQGAP4jDOw='
                                            ),
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNo7rpBvz8DgYN9uYGXPL6kKZCvYvwlntHdNYrDnUGawn8PIYB'
                                                '8CxDwMDPacUMwAxfOaVMF65YFsE8UZdiA2AGh7Ddc='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'graphs',
                                    [
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//RpbnvO2Em5qmRR9OypDiOtB6vJxQMZ0GSIqDCZEMB'
                                                '5eHuxNAS1CPH8piDgMqXbOUuXqkjwWLiZ7n5Z2QKi2OqoFIgIL'
                                                'sLyt0D2+16tSPRXsMEEUACymyWEQ6uOafxO+6O/xWGnUEsd5f4'
                                                'kR/a1bTDSnfwl8hAvW55Zpgc/pZ7Q9rsuZVP+jjlWvQlwDqO7U'
                                                'NcdPXBpl0OFtjKurVcmjgdEJpoH9JZDxXx+7mKa2ZR2NV0IMNp'
                                                '/4SEuS8O0LYIHcbPsfMkpkx3xTeMnFcnqALcAMa0Ez4ZHnOwKu'
                                                'pAHL6RfDeIRCGcnuseI/9mqVxStRZ6jxsMDs8O3a5chfolyVnr'
                                                'iww2en128k1Ojl7/5++RKA9xvEN5Dx5K0ErM89cHp0iIzSUu50'
                                                '/GbIzwONzkeVSs4jrVhnQ6rVDMmYksINLbkwFFeqYdCHJycaUI'
                                                'TiqfKYPVDPiTsd4Vbs0zEi6Ot2yGWhi2o2cBp8Y5ZdilN98s75'
                                                'ARWaMHQcN4Bw2jAHtcyN1RlG3CgWDGcvJ6BLyc8Z2H7SwHFCrz'
                                                'oZmTU8bdz4TkHaEL7faVF+6vTH6JJaPbxNjeagGSfUn/hLLk4t'
                                                'hW60wpvF2bWe3WeB2YwKZdRAl1NsYFPKuQpXThoQbd42pcyV3g'
                                                'AwxsL03JP6rTUrzxGoOgKATToWMdBZfqAN96o31GPePHoJBfaP'
                                                'ORyv455/Xlv0LGVdg3D+TwDezdWht/dDHDCjBAs/hhUmwUCsMV'
                                                'I9j+lEV99laJ8E2DJKnqpNjmpcyS2A95ajgv/9KaZy68/gKRe6'
                                                'ciidt4sLUk+EIUpVd5kQxaVYErhJy8pS8xwpi3ZxAabTaU07GU'
                                                'k8vUIU/pQWNSsqrDqk'
                                            ),
                                        },
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//YzSUu50/GbIzwONzkeVSs4jrVhnQ6rVDMmYksINLb'
                                                'kweIRCGcnuseI/9mqVxStRZ6jxsMDs8O3a5chfolyVnriuy5lU'
                                                '/6OOVa9CXAOo7tQ1x09cGmXQ4W2Mq6tVyaOB0aGLajZwGnxjll'
                                                '2KU33yzvkBFZowdBw3gHDaMAe1zI3VowQLP4YVJsFArDFSPY/p'
                                                'RFffZWifBNgySp6qTY5qXMkUV6ph0IcnJxpQhOKp8pg9UM+JOx'
                                                '3hVuzTMSLo63bIZfsfMkpkx3xTeMnFcnqALcAMa0Ez4ZHnOwKu'
                                                'pAHL6RfDADDGwvTck/qtNSvPEag6AoBNOhYx0Fl+oA33qjfUY9'
                                                '4FIgILsLyt0D2+16tSPRXsMEEUACymyWEQ6uOafxO+6B7sTQEt'
                                                'Qjx/KYg4DKl2zlLl6pI8Fi4me5+WdkCotjqqCaaB/SWQ8V8fu5'
                                                'imtmUdjVdCDDaf+EhLkvDtC2CB3Gw8egkF9o85HK/jnn9eW/Qs'
                                                'ZV2DcP5PAN7N1aG390McMBlG3CgWDGcvJ6BLyc8Z2H7SwHFCrz'
                                                'oZmTU8bdz4TkHaxaVYErhJy8pS8xwpi3ZxAabTaU07GUk8vUIU'
                                                '/pQWNSvF2bWe3WeB2YwKZdRAl1NsYFPKuQpXThoQbd42pcyV3q'
                                                'flZKMxNFYd2AOSvb5foxqfh2gpKzkhWxhYxYjJRTdz47DEQpj8'
                                                'HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFUU9lLa5ARYUxwVnt'
                                                'Xs9BtVdkDqCQ7LRbewIa3rj8lsScPXUHWipvXQCVdEs2REehcQ'
                                                '/ghn1UCX9VjqyIUlxYsw47DEQpj8HBSa+/TImW+5JCeuQeRkm5'
                                                'NMpJWZG3hSuFV24TjA'
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'invented',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYMAEAAAUAAE=',
                                        },
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYEAFjEAEAAAZAAM=',
                                        },
                                    ],
                                ],
                                [
                                    'populations',
                                    [
                                        (
                                            '6fba85ebaa5734e567c55e9b2574cc795d717e9939d24d6ceafb5b'
                                            '2d8450db80'
                                        ),
                                        (
                                            '355218f8c5cdbc86dcbbd593c9717819093d592c356396f0ee3cfe'
                                            '87a72d5704'
                                        ),
                                    ],
                                ],
                                [
                                    'reported',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNprYGBg8AFifyDOBWI+ILYA4gQonwOIFYFYHYh5gVgAqtYJiN'
                                                'mA2BOIxYA4FIitgRgA12AEjw=='
                                            ),
                                        },
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpTZGBgyAViPiDmBWIxIFYH4gQgZgNifyD2AWILIPYEYgEgtg'
                                                'ZiJyDmZoAAIyCug7IBm+gDsQ=='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'stable',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjZMQEAADmABU=',
                                        },
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjZMQEAADmABU=',
                                        },
                                    ],
                                ],
                            ],
                        },
                    },
                    '4900ec920ed1ddbb2cc84161b74aa2c6eec1d5b49e686ae477a8b8fda22334a4': {
                        'tests': ['test_node_list.py::TestAgainstLiveRN::test_neighbour_scale'],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'max_depth': 2,
                                            'band_threshold': 0.3,
                                            'max_weight': 3.0,
                                            'recurrence': {
                                                'enabled': True,
                                                'allow_hidden_to_hidden': True,
                                                'iteration_level': 2,
                                                'allow_backward': True,
                                                'allow_lateral': True,
                                                'allow_self_loops': True,
                                                'activate_time': 20,
                                            },
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            1,
                            'configured',
                            1e-06,
                        ],
                        'value': {
                            'dict': [
                                [
                                    'connections',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNrbz8DAEAjEGUCcDsQsQLwIiHcB8VoglgBiJiD2BGIuIC4A4j'
                                                'wgbgJiViBezAABW4G4G4gBScsHog=='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'fitness',
                                    [
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNo7Ec9k/+ySgf2laGn7XHV2+zIpe/saTRv7qg3K9v+um9l/c2'
                                                'K0P2plZ//rmI09W6SivcTlQ3YmQPmaPaL2PJOF7RUSGOwZGBzs'
                                                'v/z7b6fFwGAPAKcvG/k='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'graphs',
                                    [
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//ehshYRbKdOvlE66gqtBvSwmQI0f9S+1kUP85QL4Br'
                                                '8wc+2sOMgJF3wMI6yGUc/erks/cO8NxZV0kF44KXDE8XmONM+P'
                                                'gqRa1NMYQ0etegp2W0xvdmqcaYmGeF7f3Hf2zDLAfNlIBDVeyH'
                                                'z8yhdk02kFAkux0AYtKzXxc8CoNWFI2A8ToF8ZRZ1iyycbndaB'
                                                'v9IOiJiVPpD1FrgFBMHJhujD8o/6IoL7VCeK/qXrSdfN/7Ttb5'
                                                'sQQgW1NGJxa/52NomhUjZt/N7mXw9Ddwduyq1c7pud+WZUMIId'
                                                'ufr3vsVEqkytI4IOLp9x8W+wB/bxmFs6bwaRQC1JHbqznkSB9N'
                                                'O00m5Fa4kmis6ZvmdtwRZwGLdfcyIUZQXzVDOKWLqD2Ppyoh7V'
                                                'VaZYD8cMjE5dCmyUJCwakVg57hT3B636g98X5Z4FR5yw59P0cZ'
                                                'nCDqKYmW1e3080xHuHme54zYxGfdnQg/2XE1WR70mUoCwoR4KM'
                                                '5olOI0PviNR+ExCir3fGm5bq9fywjr8R1U9jgZwZXbmAT/2lrv'
                                                'gDp6rpYE404n+jWKFYPc4j6Sd1J2h8epJUV/GSyzUC1nKM037V'
                                                'aK38hhEBCnUq/6DkmHnSuel6WD+RldTWv+bU9ZO8hoxil93F00'
                                                'h1JD++XaUL5Kqt+5ioQsGXEee373DuyTCMkyEsH6WqcYKtkNpa'
                                                'xlZ3Z2DaNqc5E4fI0gIDBDxq76Ge85tuLj/jsMRCmPwcFJr79M'
                                                'iZb7kkJ65B5GSbk0yklZkbeFK4VQvDB7kWOM8dgzeiWhki5vI0'
                                                'KelOrHEiPTK4ShVMy/QcsC6sci9sM+0tI0eB1YD02CPp5m34Aq'
                                                '8XDYx1DNe1DpPa4kMG'
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'invented',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYGBgYGSEYAYoBgAAVAAH',
                                        },
                                    ],
                                ],
                                [
                                    'populations',
                                    [
                                        (
                                            '6fba85ebaa5734e567c55e9b2574cc795d717e9939d24d6ceafb5b'
                                            '2d8450db80'
                                        ),
                                    ],
                                ],
                                [
                                    'reported',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNrbz8DAEAjEGUCcDsQsQLwIiHcB8VoglgBiJiD2BGIuIC4A4j'
                                                'wgbgJiViBezAABW4G4G4gBScsHog=='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'stable',
                                    [{'array': '|b1', 'shape': [20], 'zlib64': 'eNpjZMQEAADmABU='}],
                                ],
                            ],
                        },
                    },
                    '497cf110419b366d5ebc27b8c165a932fea4dcc2aa851d893516fc1149442ceb': {
                        'tests': ['test_node_list.py::TestAgainstLiveRN::test_neighbour_scale'],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'max_depth': 2,
                                            'band_threshold': 0.3,
                                            'max_weight': 3.0,
                                            'recurrence': {
                                                'enabled': True,
                                                'allow_hidden_to_hidden': True,
                                                'iteration_level': 2,
                                                'allow_backward': True,
                                                'allow_lateral': True,
                                                'allow_self_loops': True,
                                                'activate_time': 20,
                                            },
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            1,
                            'rn',
                            1e-06,
                        ],
                        'value': {
                            'dict': [
                                [
                                    'connections',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNoTZ2RgUABiMyAWAmIHIFYG4s8MDAwSQNoQiO2BWBoqpwfEBk'
                                                'CsCKUZgFgLiNWBmAuIAbCdA9Y='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'fitness',
                                    [
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjaHhn/sNY2v77n3r7LAYGewYovvDQ2H4DE4LPA6UZ0WgGND'
                                                '4A53oNGg=='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'graphs',
                                    [
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//W8vCGS6YwD8trq8UwvEOcFl+ZdFur1Q6M9oHtH3ME'
                                                'b1W8hEeMibRD9kumMTq9PJNvNGUo5FGFmM/Dnqpzd1leato+Uk'
                                                'hNI4iOn7/YWj2+W797xZUmpvj6jg2nDyDDu4jGCQh5s/GqoRxP'
                                                'aMulBRbsBdvWa0uX4lMRYk16LjWQRPkHCgv8FfK77Wam2AQxfY'
                                                'e+SQag/ygb+tS7nxuGmpjqLae6794wKyhKMJGki479ckP+RCHa'
                                                'nYhsi+YBejGrm15gdfmq5D+xA+QaeqkCteCkUPm4Uqb6ysbRPj'
                                                'KIKyPT0gAvlbnsDDosEGq986s1BuyCS+Q9vfgKL4pwVqsNgrRd'
                                                'zgMDl6QRlfvjkFIOBKH4S6WJfxUngNMsnw6yZJa3u+cmJW6uy8'
                                                '/xV+TWo5qP6SoSstW3jJNAXKGwMq3EMa1cCXqMejArVBUoPorf'
                                                'hbiuxPk19tZx5SNVVHcP1ECNNvEnshjcz4zI04FUp86fW8nUO2'
                                                'vd3vydM6irklqcDH7joceMEf18Nb/WxLxxuQZcikGvhVv4IY4s'
                                                '1ONgQwwgQEM8LTz+//eGNIWvPBC6Xh25qQ/mpxeWf6jNH01lYx'
                                                'tc20V87460y8y+RnAywJOhXcZmVPtGRWLt+aWCTUQVPF8xUll0'
                                                'TAYRA6VbTO+7ac7no8Q16ia9H/i/cLwBND+SAQRcBaXcReM76G'
                                                'XeUTc8yEUY7U+ZcFgUUnV+2A5lHKH2nLka25VfMZy0NR1pVF6h'
                                                'rLRyEn5qrB9DKiv44FfueNKGmHtmoGiWgWpFr5P8Fk1AGYIHL6'
                                                'gIJ/MZ6KZc3TzNlH6G3xEKkvieYZPnxgZB5rr3uVB/qvIARciK'
                                                'HDtRiEQEwjGn6DpEci'
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'invented',
                                    [{'array': '|b1', 'shape': [20], 'zlib64': 'eNpjYMAEAAAUAAE='}],
                                ],
                                [
                                    'populations',
                                    [
                                        (
                                            '6fba85ebaa5734e567c55e9b2574cc795d717e9939d24d6ceafb5b'
                                            '2d8450db80'
                                        ),
                                    ],
                                ],
                                [
                                    'reported',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNoTZ2RgUABiMyAWAmIHIFYG4s8MDAwSQNoQiO2BWBoqpwfEBk'
                                                'CsCKUZgFgLiNWBmAuIAbCdA9Y='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'stable',
                                    [{'array': '|b1', 'shape': [20], 'zlib64': 'eNpjZMQEAADmABU='}],
                                ],
                            ],
                        },
                    },
                    '6c14dfb1e38a30c25446524a67f4f59980d8d0e3f699c28a4efa51d0ffe0cf01': {
                        'tests': [
                            (
                                'test_node_list.py::TestAgainstLiveRN::'
                                'test_rn_populations[2-with_backward-2-None]'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'max_depth': 2,
                                            'band_threshold': 0.3,
                                            'max_weight': 8.0,
                                            'recurrence': {
                                                'enabled': True,
                                                'allow_hidden_to_hidden': True,
                                                'iteration_level': 2,
                                                'allow_backward': True,
                                                'allow_lateral': False,
                                                'allow_self_loops': False,
                                            },
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            2,
                            'rn',
                            1e-06,
                        ],
                        'value': {
                            'dict': [
                                [
                                    'connections',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNrbycDAMAeIq4C4CYhFgHgVEO8A4k1AzAHEZkCcCMQCQNwAxM'
                                                'uBeBEQKwDxMSA2BuI1QLwOiAGNlwll'
                                            ),
                                        },
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNrbycDAYAbEO4DYGIhFgFgAiBcBcSIQrwLiY0C8BojXAbECEC'
                                                '8HYg4gZgXiDCBeAsRHoPoAbJ8IgA=='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'fitness',
                                    [
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNoT4nSyT/n/306GtcEuhqHBjoFBwZ6HgcE+pEHfngFIQ7CBvX'
                                                'Msm33be0n7gNN/7UBik65IguVYwPIK9kxQtQDzDBB1'
                                            ),
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNoT4nSyZ2AwsA9p0AfSCmDc9l7SftIVSXvnWDZ7HgYGexYgZg'
                                                'JiBjS8KxrB3pL83Q5EAwA5mg2k'
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'graphs',
                                    [
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//U4uZZ4ZOymRFQkm2W8Ay3WR78wEdqN9wcC/idF7rv'
                                                '9bqYebBY/78Vp+vhznaZtdDHzLV/ZE5s3R5eTdJnYr8PKivuO0'
                                                'tKLr9riwkKqyQ5jcX7y3LNg95RdpZaLA35RRkaeRIBAko+3vNa'
                                                'nxQ3CnDzOVgQON/wevds36WC+fWQZb1LwP6ltNK9yxmrlFMvZD'
                                                'N1siTSQusfY9LnbfB52QI9IrKXuEvzpCOK1qVlop9lguGam2vX'
                                                'ER/4M1gXeNLQ5wIP/cGd+EBcm3M+1MlrNydtIvben7qpouYIU4'
                                                'uswvFLiN5dC0x1xYEQflfxVNe3ppEolySm5l9+zb5dJ3U9HYOM'
                                                'Oww2en128k1Ojl7/5++RKA9xvEN5Dx5K0ErM89cHp0iBz2RzqM'
                                                'TfxXVPQXA9s/ZpdwBpCMNmAcDYd21NpBy+oWl1ZKzUsLZ91Pyi'
                                                'dOFiqJXwy7+EO8kDigX+m7DItymPR16mXsoia5O5BXkbyYLc14'
                                                'gFHE3RKHg1VBLqqrfYpuI95hh9KxJaou2c4yuPd0kw9uPRUrL6'
                                                'RX9iHyA6utWsGk3Kqg68sKnt3SieXlweXCRThQLI6vttaDswk8'
                                                'PfLRvifFXlaDP65PH2WArFonNyjURfMyD9Dum8BYTpiI0TYj/R'
                                                'XmxC2aACGKswaPDJjKfGbNJOj5E5ezDmwqXJlIWVZDSNo7PTf0'
                                                'y2nZwKkEvRxZHZEe7F3sr0uB+1ObJ+Hg6mir20euIM4/XFs7df'
                                                'qlC1I0BITGYXn30G5uXPi+1Ddgi2RyXFNTEdBj/KlDK1g7LGdt'
                                                '0oU01UJ6UL78k9efedcTTqrtStXTvxCeq/SaIcwib4z0tHskQD'
                                                'SxvF+OEopnKYDS70TF'
                                            ),
                                        },
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//U4uZZ4ZOymRFQkm2W8Ay3WR78wEdqN9wcC/idF7rv'
                                                '9bHPZHOoxN/FdU9BcD2z9ml3AGkIw2YBwNh3bU2kHL6hb/3Bnf'
                                                'hAXJtzPtTJazcnbSL23p+6qaLmCFOLrMLxS4javbR64gzj9cWz'
                                                't1+qULUjQEhMZheffQbm5c+L7UN2CL1LwP6ltNK9yxmrlFMvZD'
                                                'N1siTSQusfY9LnbfB52QI9J16mXsoia5O5BXkbyYLc14gFHE3R'
                                                'KHg1VBLqqrfYpuI8VeVoM/rk8fZYCsWic3KNRF8zIP0O6bwFhO'
                                                'mIjRNiP9l1ZKzUsLZ91PyidOFiqJXwy7+EO8kDigX+m7DItymP'
                                                'QrKXuEvzpCOK1qVlop9lguGam2vXER/4M1gXeNLQ5wIEjaOz03'
                                                '9Mtp2cCpBL0cWR2RHuxd7K9LgftTmyfh4OpoZHJcU1MR0GP8qU'
                                                'MrWDssZ23ShTTVQnpQvvyT15951xNOqu1K1dO/EJ6r9JohzCJv'
                                                'jPS0eyRANLG8X44SimcpgBXmxC2aACGKswaPDJjKfGbNJOj5E5'
                                                'ezDmwqXJlIWVZD3Kqg68sKnt3SieXlweXCRThQLI6vttaDswk8'
                                                'PfLRvieww2en128k1Ojl7/5++RKA9xvEN5Dx5K0ErM89cHp0iK'
                                                'xgovw/EbCCeVwH88HA3a5FOPqv4bi6DD95Kzq7yb2H0acrbCRe'
                                                'bM2A9FPe/esgIijs5K5c0VfMbapARFmYB8V9v7/LRojCtcNN24'
                                                '/kAw5xY8wAi4V34gVigX0Q/u+XGobekQHC9aa/HSjQlligzldn'
                                                'jHMNunUtm2inYg7TdWnwGEfdeNLuLq4zweUxtzj9bWMtc4tomf'
                                                'QQxXp3lv3fOfqklEK4'
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'invented',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYMAEAAAUAAE=',
                                        },
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYMAEAAAUAAE=',
                                        },
                                    ],
                                ],
                                [
                                    'populations',
                                    [
                                        (
                                            '6fba85ebaa5734e567c55e9b2574cc795d717e9939d24d6ceafb5b'
                                            '2d8450db80'
                                        ),
                                        (
                                            '0dbc2fc6dd354a64636de88a1c4334670d6a93dd6b97799479d1ab'
                                            '99776c966f'
                                        ),
                                    ],
                                ],
                                [
                                    'reported',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNrbycDAMAeIq4C4CYhFgHgVEO8A4k1AzAHEZkCcCMQCQNwAxM'
                                                'uBeBEQKwDxMSA2BuI1QLwOiAGNlwll'
                                            ),
                                        },
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNrbycDAYAbEO4DYGIhFgFgAiBcBcSIQrwLiY0C8BojXAbECEC'
                                                '8HYg4gZgXiDCBeAsRHoPoAbJ8IgA=='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'stable',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjZMQEAADmABU=',
                                        },
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjZMQEAADmABU=',
                                        },
                                    ],
                                ],
                            ],
                        },
                    },
                    '71a9254d1acf88c61ad5f72af6a4e5cc2c415223631dfddf91a3ee94c433d141': {
                        'tests': [
                            (
                                'test_node_list.py::TestAgainstLiveRN::'
                                'test_rn_populations[2-full_recurrent-2-20]'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'max_depth': 2,
                                            'band_threshold': 0.3,
                                            'max_weight': 8.0,
                                            'recurrence': {
                                                'enabled': True,
                                                'allow_hidden_to_hidden': True,
                                                'iteration_level': 2,
                                                'allow_backward': True,
                                                'allow_lateral': True,
                                                'allow_self_loops': True,
                                                'activate_time': 20,
                                            },
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            2,
                            'rn',
                            1e-06,
                        ],
                        'value': {
                            'dict': [
                                [
                                    'connections',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNp7x8DAcASI5wLxeiCWAOKHQPwdiO8AMRcQOwFxJRDLAPEcIL'
                                                '4J1WMGxB+A2B2q9hoQAwD/dwwE'
                                            ),
                                        },
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNp7x8DAMAeIZYC4EoidgNgdiCWA+DsQH4Hih0D8AYivAfEdID'
                                                'YD4pNAfBqILwHxEyBOAWIA68cMpw=='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'fitness',
                                    [
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjYHCwn9rDYC/D2mDHwMBgz8CgYM8DpM8EC9tD+CBsAMc99X'
                                                'ZgsRPCTGCaDaoHphYA9d4LKA=='
                                            ),
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjYHCw76m3s2dgMEDCCmB8JljY/oQwk/3UHgZ7HgYGezYgZk'
                                                'DCt2Qb7UDqWYHsq1GMYD0AKwMNhA=='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'graphs',
                                    [
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//ZzoaqUt2GT2rsCA1qaUsyQvnBcd1/0bj5VJyPxwGL'
                                                'JH41tQqzVp6i8h6kPsswX2GdEboSVupftV3rVwWhSVCOcN/AiV'
                                                'PdL+RdjQb9JFDwVrnA4slHNcGPEsff+Vv3spOe55BXXeLnfhqe'
                                                'lJ/Wnbgt/uTBeH4OZmTYLTGdZZrwPEex+/suS5VgKYRH3UCWp/'
                                                'ORiefT0qgeagP5nq2IXJx0ji4blpPXjdPveF7Qf39HCLksmZIH'
                                                'wmbXSclLe6mApIy4uihfJaJ/uHPaNuioGdsFTgf4pOXvZkSkyS'
                                                '1470gFgBZ46HeFpp9wvu5oHaTHbUmPW4j9DPrsMfoVglQW6ZrQ'
                                                'JB4yuRjPcH4tmEsZdZmQ4VAmkcF5iEM+JUl4cRCUjrjpaLhpsV'
                                                'Ax11HAwLwED/K58tGnFzsoP8H51otqbhPDlwSN5McfkSjG3hr2'
                                                'hEYr1jqbXPKX/FXKlfioP9OT4UV5iSfzYfQt7LFwotKYbxXBUn'
                                                'h6w9k6i9BtSdSA1M02ISfx4dc8YIj1nmoL9EUbWkyAOIOm7EFU'
                                                'DNhahMiDh3iuCeSFi/3HJ2LczGER8pDVsOLyWeAJBcQa13Z2sF'
                                                'piM7OG38EpUKkjhwF+Sm+zsh1pWS/GEph5vCQXtLPx8v0I7/gZ'
                                                'JjUPiLog3fqomj2ktXhQk1WUSjPhtBCFueeMr7t23/rOi7xyes'
                                                'Ci0DLckN79PgpTaIz1WWUKY2jx77jBWHr/jQsvuUI6C712V9Pa'
                                                'pPQ6CvviwyiOpGRlrz45XTFpgG9wAih6w/9eMX93rw5v3H3k6v'
                                                'H5RkJelx9MhSB+eKasytjp/JRZ5OoyR+SQaxJoTqDX68/Z6l67'
                                                'Gs9KYQwahu52L4ZT+o'
                                            ),
                                        },
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//ZzoaqUt2GT2rsCA1qaUsyQvnBcd1/0bj5VJyPxwGL'
                                                'JHHh1zxgiPWeagv0RRtaTIA4g6bsQVQM2FqEyIOHeK4J6SfzYf'
                                                'Qt7LFwotKYbxXBUnh6w9k6i9BtSdSA1M02ISf0jeTHH5Eoxt4a'
                                                '9oRGK9Y6m1zyl/xVypX4qD/Tk+FFeYlouGmxUDHXUcDAvAQP8r'
                                                'ny0acXOyg/wfnWi2puE8OXDQsvuUI6C712V9PapPQ6CvviwyiO'
                                                'pGRlrz45XTFpgG93sfv7LkuVYCmER91AlqfzkYnn09KoHmoD+Z'
                                                '6tiFycdIi6KF8lon+4c9o26KgZ2wVOB/ik5e9mRKTJLXjvSAWA'
                                                'H8EpUKkjhwF+Sm+zsh1pWS/GEph5vCQXtLPx8v0I7/geNbUKs1'
                                                'aeovIepD7LMF9hnRG6ElbqX7Vd61cFoUlQjn4uG5aT143T73he'
                                                '0H9/Rwi5LJmSB8Jm10nJS3upgKSMus6LvHJ6wKLQMtyQ3v0+Cl'
                                                'NojPVZZQpjaPHvuMFYev+I6fyUWeTqMkfkkGsSaE6g1+vP2epe'
                                                'uxrPSmEMGobudiACKHrD/14xf3evDm/cfeTq8flGQl6XH0yFIH'
                                                '54pqzK2SY1D4i6IN36qJo9pLV4UJNVlEoz4bQQhbnnjK+7dt/1'
                                                '82yAI0y5OjPSEooWORsV5tGt/EB0VRm1qPImUTEafpCWfWidi/'
                                                'AiqOxuNfq0IbDccM6FM5PODCBvuePJNtEcfCYSItIaVrjwzAtu'
                                                'u7kl/IhliK37kKOLw41WeTehPywS++TOklrIEeezFYmubCkPM6'
                                                'Ern/Qp9pNy/MWLG0cLMc2IVCAmPJIhhEeuSiNIGFNeA+mtNlbi'
                                                'SllFnkoyIEekr4hD5d'
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'invented',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYMAEAAAUAAE=',
                                        },
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYMAEAAAUAAE=',
                                        },
                                    ],
                                ],
                                [
                                    'populations',
                                    [
                                        (
                                            '6fba85ebaa5734e567c55e9b2574cc795d717e9939d24d6ceafb5b'
                                            '2d8450db80'
                                        ),
                                        (
                                            'c506c5c74656b565ec9eb88362a95205d34b37774a5c04d2b78a6f'
                                            '88866abeb3'
                                        ),
                                    ],
                                ],
                                [
                                    'reported',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNp7x8DAcASI5wLxeiCWAOKHQPwdiO8AMRcQOwFxJRDLAPEcIL'
                                                '4J1WMGxB+A2B2q9hoQAwD/dwwE'
                                            ),
                                        },
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNp7x8DAMAeIZYC4EoidgNgdiCWA+DsQH4Hih0D8AYivAfEdID'
                                                'YD4pNAfBqILwHxEyBOAWIA68cMpw=='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'stable',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjZMQEAADmABU=',
                                        },
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjZMQEAADmABU=',
                                        },
                                    ],
                                ],
                            ],
                        },
                    },
                    '75590d818b916ab21199bb857c0729470700b42e71fa8f60d45101024d4296fb': {
                        'tests': [
                            (
                                'test_node_list.py::TestAgainstLiveRN::'
                                'test_depth_three[full_recurrent-20]'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'max_depth': 3,
                                            'band_threshold': 0.3,
                                            'max_weight': 8.0,
                                            'recurrence': {
                                                'enabled': True,
                                                'allow_hidden_to_hidden': True,
                                                'iteration_level': 2,
                                                'allow_backward': True,
                                                'allow_lateral': True,
                                                'allow_self_loops': True,
                                                'activate_time': 20,
                                            },
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            2,
                            'rn',
                            1e-06,
                        ],
                        'value': {
                            'dict': [
                                [
                                    'connections',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNp7xcXA0MzBwODAzsDwmY2BQZOBgWEpUCyAB8jgZmBYCaRsgF'
                                                'iZlYHhFZDmAqr9BsQvgfg8EwNDFlANCBQB9fRwMjAAALjCCk4='
                                            ),
                                        },
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjYGBg+MbBwPAKSDcDaWVWBoYAHgYGGyBfE4gZuBkYXgLFuY'
                                                'B4KRcDw2c2BoYeTgaGIiCbBSg9EyjuBGQcYWZg2ABkAwB/xQmO'
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'fitness',
                                    [
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjYGCwX/hf377p1W87NiCbgUHB/osXg/3uIEV7rxfyQD5ErP'
                                                '+9PpA2sK8NBPEd7J/nMULlIHwYGwBaJRGm'
                                            ),
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjYHCwZwBjA/uF//Xt+9/r2+8OUgTyFcDY64W8/fM8RvvaQA'
                                                'b7L14M9mwMDEBxCC6Tsoeqg/BvijPYAwAEwxEf'
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'graphs',
                                    [
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//XqUZ4Rxoinj2oCsQ3spXyc+d8u12IMlllWr2o/lIJ'
                                                'GPTbP+tEx/Rx1sJCxOnUcY++hzFbSIHgq+mI/JXvp7O/F6HNSx'
                                                'tkaMAKsJmU/SyA+RtlDj+ccnQW4eLWmXnC0M2uMRlgHThGNzeU'
                                                'EE9azZYXCMIDD0XMXi+ulYKyCQFmXMFzoO1EplkeqXMiIDzEsF'
                                                'V5TN9l+iV05VGzTHpVt/Q/uu3lAadOO2ZUZrTWFcQiySvBdCTW'
                                                'lC24yxdaa+b5FWyATRqdJ4Ji2QDMDCIkaAtj+oJk8weIFmSSjk'
                                                'PNZNuIqdjGTN/BwIc3zefGUVvIbOFkIZS9vONU3tU/cicq+MQy'
                                                'odsI88qWOxaU0bZYw6QEy6iaYhYodYgAst/XkdU1duEOzvdRDy'
                                                'zGxuZVzeLDECqQI9TlOFbGBi3j0QPS7rNxsNwdypZOIHC1a7Qf'
                                                'NkphoHa0o6e0zAi7oadT+ZTBycxnhvcfUaijXhoUqEX8Z8OwLe'
                                                'V4gwaToVYGan31mzuz0ao3+eElGao27PGcWhyVNYjLBUP3YlOK'
                                                'yWonRSNwoZF3ctPvwr26PHyS6jqLa52pLiNj4/iJzw/ZYtrKTq'
                                                'RftWcL+s1i3B75CghT3ukjTrCvN4jPKs9AonVrxFC3/TIVUKL9'
                                                'ET01hH/nbmyjGDx0P7usFlWPE5QdOP6+rRUpwQT9bsUzTxUN9S'
                                                'HbSV350iHr0pnuPyWz9cV2qrc+AhLYRw9mnjsMRCmPwcFJr79M'
                                                'iZb7kkJ65B5GSbk0yklZkbeFK4VQchBbfnnFYbceVJ4C3KY7H6'
                                                '/Gj6sv0ZQyG3f7pTRjucx1sOhZX/WVt6xi4QTDTXWsVQWfXe4V'
                                                'oX8pG6Yzgxy21U6TSo'
                                            ),
                                        },
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//eOwxEKY/BwUmvv0yJlvuSQnrkHkZJuTTKSVmRt4Ur'
                                                'hVPvwr26PHyS6jqLa52pLiNj4/iJzw/ZYtrKTqRftWcL9vcfUa'
                                                'ijXhoUqEX8Z8OwLeV4gwaToVYGan31mzuz0ao02z/rRMf0cdbC'
                                                'QsTp1HGPvocxW0iB4KvpiPyV76ezvxwdypZOIHC1a7QfNkphoH'
                                                'a0o6e0zAi7oadT+ZTBycxngE0anSeCYtkAzAwiJGgLY/qCZPMH'
                                                'iBZkko5DzWTbiKnezvdRDyzGxuZVzeLDECqQI9TlOFbGBi3j0Q'
                                                'PS7rNxsNFzoO1EplkeqXMiIDzEsFV5TN9l+iV05VGzTHpVt/Q/'
                                                'uMZM38HAhzfN58ZRW8hs4WQhlL2841Te1T9yJyr4xDKqzWLcHv'
                                                'kKCFPe6SNOsK83iM8qz0CidWvEULf9MhVQovf54SUZqjbs8Zxa'
                                                'HJU1iMsFQ/diU4rJaidFI3ChkXdy2u3lAadOO2ZUZrTWFcQiyS'
                                                'vBdCTWlC24yxdaa+b5FWyOMRlgHThGNzeUEE9azZYXCMIDD0XM'
                                                'Xi+ulYKyCQFmXMx1sOhZX/WVt6xi4QTDTXWsVQWfXe4VoX8pG6'
                                                'Yzgxy20HIQW355xWG3HlSeAtymOx+vxo+rL9GUMht3+6U0Y7nN'
                                                'gPE6BfGUWdYssnG53Wgb/SDoiYlT6Q9Ra4BQTByYbohks5PL/x'
                                                'pNjFzEjzIs/M92pDme5aB9auYl0SfGifUUR52VlHhzF49uh9Cq'
                                                'jgjzR8Bo/lnFOxZosGz9+CGpt6PZS3d76ihsw1Si+9DuhPB4/p'
                                                'O2e2zSsUlajbAKSEDeByBleK3STLa3P2/Eb+zPQrRWctXPW9eQ'
                                                '+HK4ZrN52neSBqQDWq'
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'invented',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYEACjBAKAAAbAAI=',
                                        },
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYGRABowgBAAALgAE',
                                        },
                                    ],
                                ],
                                [
                                    'populations',
                                    [
                                        (
                                            '6fba85ebaa5734e567c55e9b2574cc795d717e9939d24d6ceafb5b'
                                            '2d8450db80'
                                        ),
                                        (
                                            '178a294edb0a40640c8b715bc1e723166ec6cf39b51450a723fc7e'
                                            '4feea65374'
                                        ),
                                    ],
                                ],
                                [
                                    'reported',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNp7xcXA0MzBwODAzsDwmY2BQZOBgWEpUCyAB8jgZmBYCaRsgF'
                                                'iZlYHhFZDmAqr9BsQvgfg8EwNDFlANCBQB9fRwMjAAALjCCk4='
                                            ),
                                        },
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjYGBg+MbBwPAKSDcDaWVWBoYAHgYGGyBfE4gZuBkYXgLFuY'
                                                'B4KRcDw2c2BoYeTgaGIiCbBSg9EyjuBGQcYWZg2ABkAwB/xQmO'
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'stable',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYGBgYARhRjAF5gAAAEcABg==',
                                        },
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjZGBkYGBgBBEgAKYBAFoABg==',
                                        },
                                    ],
                                ],
                            ],
                        },
                    },
                    '7d283bbf0f2b2ee684a81ea4ee090e94889a96cf25614d8043ecd047ce6e7c43': {
                        'tests': [
                            (
                                'test_node_list.py::TestAgainstLiveRN::'
                                'test_rn_populations[2-hidden_only-0-None]'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'max_depth': 2,
                                            'band_threshold': 0.3,
                                            'max_weight': 8.0,
                                            'recurrence': {
                                                'enabled': True,
                                                'allow_hidden_to_hidden': True,
                                                'iteration_level': 0,
                                                'allow_backward': False,
                                                'allow_lateral': False,
                                                'allow_self_loops': False,
                                            },
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            2,
                            'rn',
                            1e-06,
                        ],
                        'value': {
                            'dict': [
                                [
                                    'connections',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNrTZWBgkANiYSDWA2IOIGYFYgkoHwSEgFgEiDmhciA14kDMBM'
                                                'QqQMwHxPJQNgBKWAGq'
                                            ),
                                        },
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjYGBg0ANiViAWAmJOIOYDYg4gFgZicSBmgspLQMXlgFgFiN'
                                                'UYIAAkJg/EkkAMADR4AV4='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'fitness',
                                    [
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjYGCwZwbiBlYhewYgzcCgYM/CYGAvDWT/+W8P5DsAsYG9QP'
                                                'd3uyf/9e3VgOICQLxoMbM9hzKDPRNUD0Qvgz0AOUUODg=='
                                            ),
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjYHCw//Pf3p6FwcCeAYif/NcH0gpg3MAqZL9oMbM9hzKDvR'
                                                'oDg700EAsAMTMQMwGxirU5UJ2D/dv/DECawf5bF789AMBvEM4='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'graphs',
                                    [
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//b4iBli6mBW5y2wajEJLaCfj5Nzbfnt9zMsbKeZfU9'
                                                'QwJzJopyzjFPmJ4YSqbhUTNt4AYqfDhlaEabXoIDgyjUbNdLO5'
                                                'Qy68KoJVhpIi0a4oPiuZ+8L4sz1tSjrBERnfGLWRy2z6br1SRG'
                                                'sT+9p0KH51xQwPF/HFrffBvl8GJCqwPr9qx/B0XuPiWChx887i'
                                                'eqHseVBzMYtqZW0s6wrxpiasBlm4bYiXUPm0NYHPSC4a8ioKtl'
                                                'XAm1GM818XC1uDgNxnMq3y3RusagmWRZHDFh3uG2nwp2GjgoFG'
                                                'offGpwrIjOU17ogBvJqmZfkeKitmdd9DK6Rff3Ioe2mtO+cJ8a'
                                                'vjsMRCmPwcFJr79MiZb7kkJ65B5GSbk0yklZkbeFK4VdBmQA5y'
                                                'eBdMwGWHv78gJM2mfVBc6duP4JTb9wECq1aFU1Uyp8vcwahhn0'
                                                'TsCtI1sT8gc9X86X5mbpq40/KB5ZJh3oeUuEDG4QUcCAH3/LWZ'
                                                'Nmaci/Dc0NAJgYE+1Kpw2J+iqDNPIZzIztx79R5XYYFQ11g8Cr'
                                                'fq7iacoy3qK97VIs8lhwq5p4Lz2jCL4rSFpVsb64eAipTHsBH4'
                                                '8o3nhHHBeAq4DwBO95JPMFskTggt3sn7IHE1+408Y7+F71Tbxo'
                                                'lk80OEh6VbDk3mWU4YVHlAMI3KPwP4VGBnN3I5ydgSBazc3Vw4'
                                                'BZD48xDjPfFr1e93XK4F8sV5rGhX8NKdfqQds+PY31/WXH6sdG'
                                                'vhTWra+2kxpFO5Wr9sP3+pY2f3jHDlE7X3m5UmVp12YePialWR'
                                                'yl8Fwt0qO07E6sq/BOiM6gXkCRLFQad9cQQmw6Mj//Ud9SibTZ'
                                                'UFmLs3tnhuuTC7E0gD'
                                            ),
                                        },
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//eOwxEKY/BwUmvv0yJlvuSQnrkHkZJuTTKSVmRt4Ur'
                                                'hVjOU17ogBvJqmZfkeKitmdd9DK6Rff3Ioe2mtO+cJ8ausBlm4'
                                                'bYiXUPm0NYHPSC4a8ioKtlXAm1GM818XC1uDgNBmQA5yeBdMwG'
                                                'WHv78gJM2mfVBc6duP4JTb9wECq1aFYd6HlLhAxuEFHAgB9/y1'
                                                'mTZmnIvw3NDQCYGBPtSqcNgds+PY31/WXH6sdGvhTWra+2kxpF'
                                                'O5Wr9sP3+pY2f3jD6/asfwdF7j4lgocfPO4nqh7HlQczGLamVt'
                                                'LOsK8aYmzXSzuUMuvCqCVYaSItGuKD4rmfvC+LM9bUo6wREZ3x'
                                                'jBeAq4DwBO95JPMFskTggt3sn7IHE1+408Y7+F71Tbxolk80OE'
                                                'h6VbDk3mWU4YVHlAMI3KPwP4VGBnN3I5ydgSn6KoM08hnMjO3H'
                                                'v1HldhgVDXWDwKt+ruJpyjLeor3tXcZzKt8t0brGoJlkWRwxYd'
                                                '7htp8Kdho4KBRqH3xqcKyCLPJYcKuaeC89owi+K0haVbG+uHgI'
                                                'qUx7AR+PKN54RxJzJopyzjFPmJ4YSqbhUTNt4AYqfDhlaEabXo'
                                                'IDgyjUYFrNzdXDgFkPjzEOM98WvV73dcrgXyxXmsaFfw0p1+pI'
                                                'oVYXF7PFItB8CssUW3PneWGFo76dQR4XpJ44cslnI847DEQpj8'
                                                'HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFVNFW6J7/Opw7mzdr'
                                                'HCirIOtyKNosgmnYPr5jDTob2Zr6OzkWDRfm/GUhdU9w+Is4ZG'
                                                'nkQr63B8ow7tRxds2VekKHlndsj2ujV8lm+XW973GlvEhloHCL'
                                                'H+3BGPJoKXr6iNSUcy'
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'invented',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYMAEAAAUAAE=',
                                        },
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYEAFjEAMAAAYAAI=',
                                        },
                                    ],
                                ],
                                [
                                    'populations',
                                    [
                                        (
                                            '6fba85ebaa5734e567c55e9b2574cc795d717e9939d24d6ceafb5b'
                                            '2d8450db80'
                                        ),
                                        (
                                            'e6425b661e41a99af773a6e895e4dff7e134d103edb0f5bd86521f'
                                            '446acc10e8'
                                        ),
                                    ],
                                ],
                                [
                                    'reported',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNrTZWBgkANiYSDWA2IOIGYFYgkoHwSEgFgEiDmhciA14kDMBM'
                                                'QqQMwHxPJQNgBKWAGq'
                                            ),
                                        },
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjYGBg0ANiViAWAmJOIOYDYg4gFgZicSBmgspLQMXlgFgFiN'
                                                'UYIAAkJg/EkkAMADR4AV4='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'stable',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjZMQEAADmABU=',
                                        },
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjZMQEAADmABU=',
                                        },
                                    ],
                                ],
                            ],
                        },
                    },
                    '8852f167c904338f35d94733b3edb55212d4d20222d21722a9976a53f366b989': {
                        'tests': [
                            (
                                'test_node_list.py::TestAgainstLiveRN::'
                                'test_rn_populations[2-full_recurrent-1-20]'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'max_depth': 2,
                                            'band_threshold': 0.3,
                                            'max_weight': 8.0,
                                            'recurrence': {
                                                'enabled': True,
                                                'allow_hidden_to_hidden': True,
                                                'iteration_level': 1,
                                                'allow_backward': True,
                                                'allow_lateral': True,
                                                'allow_self_loops': True,
                                                'activate_time': 20,
                                            },
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            2,
                            'rn',
                            1e-06,
                        ],
                        'value': {
                            'dict': [
                                [
                                    'connections',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNp7x8DAcASI5wLxeiCWAOKHQHwYiO8wQIATEEcDsQwQ9wFxEx'
                                                'AvBGI+IP4AxAZQtaeAGADalwrV'
                                            ),
                                        },
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjYGBgeAfEfUC8EIhlgDgaiJ2A2ACIJYD4CBDzAfFDIP4AxK'
                                                'eA+A4QvwbiN0DMBMTcQLwUiAGDGwnx'
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'fitness',
                                    [
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjYHCwn9rDYC/D2mDHwMBgz8CgYM8DpMNmzoXyHYDYAI7viN'
                                                'jZg8TNHljaby1isGeD6oGoZbAHADCoDT8='
                                            ),
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjYHCwZwDiOyJ29mYPLIFsAySsAMZTexjstxYx2PMwMNizAT'
                                                'EDHDvYd5VY2pto2oD5H5ge2AEAQ1wO9Q=='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'graphs',
                                    [
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//ZzoaqUt2GT2rsCA1qaUsyQvnBcd1/0bj5VJyPxwGL'
                                                'JH41tQqzVp6i8h6kPsswX2GdEboSVupftV3rVwWhSVCOcN/AiV'
                                                'PdL+RdjQb9JFDwVrnA4slHNcGPEsff+Vv3spOe55BXXeLnfhqe'
                                                'lJ/Wnbgt/uTBeH4OZmTYLTGdZZrwPEex+/suS5VgKYRH3UCWp/'
                                                'ORiefT0qgeagP5nq2IXJx0ji4blpPXjdPveF7Qf39HCLksmZIH'
                                                'wmbXSclLe6mApIyzXaupz89bgVkto3FgND7+obyrW/w2K2/A3R'
                                                'tvienMlkZ46HeFpp9wvu5oHaTHbUmPW4j9DPrsMfoVglQW6ZrQ'
                                                'LjsMRCmPwcFJr79MiZb7kkJ65B5GSbk0yklZkbeFK4VZaLhpsV'
                                                'Ax11HAwLwED/K58tGnFzsoP8H51otqbhPDlwrlzhxmAM7dujeU'
                                                'pIp42JYuBBE3zafBVATOjFuKPmswqSfzYfQt7LFwotKYbxXBUn'
                                                'h6w9k6i9BtSdSA1M02ISfwD3uJToHhH5YCSQocFd/RkhL44Rdx'
                                                'OPjm54WTYjsieY+UCXVPO/dk7YhJLI4EvY0LqPlWih+15otg+C'
                                                'qq6yJ3EXHOjOhzJ+K6AWbi8IK3MSFo9xBH7vnMmBa9Ls0oH9Wk'
                                                'Ui/GarCJdEcVpTRkKxodBxsO6l20dq2ON1C+DsD+UErOi7xyes'
                                                'Ci0DLckN79PgpTaIz1WWUKY2jx77jBWHr/gyIwhXrYtX5SmLp4'
                                                'LG1ZPVe8Z7tMvBgGl92yWeDZJbdwAih6w/9eMX93rw5v3H3k6v'
                                                'H5RkJelx9MhSB+eKasytlmtNiU7iF5ox2PB4fba3PS+gUzpwKM'
                                                'vzc+VphKS714PjIUm6'
                                            ),
                                        },
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//eOwxEKY/BwUmvv0yJlvuSQnrkHkZJuTTKSVmRt4Ur'
                                                'hVnOhqpS3YZPauwIDWppSzJC+cFx3X/RuPlUnI/HAYskcA97iU'
                                                '6B4R+WAkkKHBXf0ZIS+OEXcTj45ueFk2I7InmBcc6M6HMn4roB'
                                                'ZuLwgrcxIWj3EEfu+cyYFr0uzSgf1akn82H0LeyxcKLSmG8VwV'
                                                'J4esPZOovQbUnUgNTNNiEn+uXOHGYAzt26N5SkinjYli4EETfN'
                                                'p8FUBM6MW4o+azCpaLhpsVAx11HAwLwED/K58tGnFzsoP8H51o'
                                                'tqbhPDlwMiMIV62LV+Upi6eCxtWT1XvGe7TLwYBpfdslng2SW3'
                                                'd7H7+y5LlWAphEfdQJan85GJ59PSqB5qA/merYhcnHSONbUKs1'
                                                'aeovIepD7LMF9hnRG6ElbqX7Vd61cFoUlQjnRSL8ZqsIl0RxWl'
                                                'NGQrGh0HGw7qXbR2rY43UL4OwP5QTi4blpPXjdPveF7Qf39HCL'
                                                'ksmZIHwmbXSclLe6mApIy6zou8cnrAotAy3JDe/T4KU2iM9Vll'
                                                'CmNo8e+4wVh6/4lmtNiU7iF5ox2PB4fba3PS+gUzpwKMvzc+Vp'
                                                'hKS714MAIoesP/XjF/d68Ob9x95Orx+UZCXpcfTIUgfnimrMre'
                                                'BrunZ8DUbKH3c2RP9EE9u2B3MvOW/RutCNdLqDR0+5A3lgAq/a'
                                                '9orhvX/ksxmpvSrbj8yca3jYoo5Q4e6Ve90wbWEekybjIiX+88'
                                                'VY96g6Jvy3rcF/l8WPXGjIgmnjtZgkz90WeNgWqVpmkbk6nGeT'
                                                'fdEBAygkS4kEgtmtzw3ckagGSKF1IrpoCl9gLMQXHUtJrGscHL'
                                                'IBrnEPkOfdqCILoECw'
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'invented',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYMAEAAAUAAE=',
                                        },
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYEADjIwMAAAZAAM=',
                                        },
                                    ],
                                ],
                                [
                                    'populations',
                                    [
                                        (
                                            '6fba85ebaa5734e567c55e9b2574cc795d717e9939d24d6ceafb5b'
                                            '2d8450db80'
                                        ),
                                        (
                                            '2e3ca6c227eb420048c3c051f99c70bd12ffeaf930492553082260'
                                            '10149c02db'
                                        ),
                                    ],
                                ],
                                [
                                    'reported',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNp7x8DAcASI5wLxeiCWAOKHQHwYiO8wQIATEEcDsQwQ9wFxEx'
                                                'AvBGI+IP4AxAZQtaeAGADalwrV'
                                            ),
                                        },
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjYGBgeAfEfUC8EIhlgDgaiJ2A2ACIJYD4CBDzAfFDIP4AxK'
                                                'eA+A4QvwbiN0DMBMTcQLwUiAGDGwnx'
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'stable',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjZMQEAADmABU=',
                                        },
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjZMQEAADmABU=',
                                        },
                                    ],
                                ],
                            ],
                        },
                    },
                },
            },
            'x86_64': {
                'research': '7440581b1',
                'recorded': '2026-10-23',
                'jax': '0.6.1',
                'jaxlib': '0.6.1',
                'numpy': '1.26.4',
                'xla_flags': ['', '--xla_cpu_multi_thread_eigen=false'],
                'values': {
                    '12b63233be3e65e553906cbcd914dcddfe412f8d5bb5944fe22970dfc9d081e3': {
                        'tests': [
                            (
                                'test_node_list.py::TestAgainstLiveRN::'
                                'test_rn_populations[2-with_self-2-None]'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'max_depth': 2,
                                            'band_threshold': 0.3,
                                            'max_weight': 8.0,
                                            'recurrence': {
                                                'enabled': True,
                                                'allow_hidden_to_hidden': True,
                                                'iteration_level': 2,
                                                'allow_backward': False,
                                                'allow_lateral': False,
                                                'allow_self_loops': True,
                                            },
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            2,
                            'rn',
                            1e-06,
                        ],
                        'value': {
                            'dict': [
                                [
                                    'connections',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNrrY2BgiATiWCCuAmIhIHYE4gIgLgNiLiDWAGI9IBYAYmEgjg'
                                                'ZibyDmAOIgIJYE4nggdgJiAPZYBTU='
                                            ),
                                        },
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNoNwyEOgDAQALAThEwuoPaDfWOzGIJigqDwKN5Pm/SLiGq2uP'
                                                'iavBweNldvu5uTp7uPsz/CJASx'
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'fitness',
                                    [
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjYGCwD2ZksJ8DxAxgoGDPAxTbGstgX/OzHiQGxAb2z8Re2T'
                                                '34r2/PCeZDMIcygz0/mK1gzwEVAwBAWg1f'
                                            ),
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNqr+Vlvz8BgYP/gvz6QVgDjrbEM9hzKDPZzGBnsg4GYn4HBng'
                                                'eIOYGYA4gZoLhMyh5MczE9sOv5X28H0gsAsf8PnA=='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'graphs',
                                    [
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//RdPHodm1yw5vbMLvd3eK4rPX/7TT/kFZMbPIF5Gl5'
                                                'ZI+M1fjR4uNuKHTIG0QKItvqnUDKwdwKgwSKmpV0/XEO1n6ueu'
                                                'YXcMxR4kMbnG6D8swWEEDMd90FHTQ4u8ifK/t+FIYaLN3e8zLQ'
                                                'ENRr00I+4UPipJMbLMM4RfMT0Ym7HUa3XdRJQmDyFlO/zYpgzt'
                                                'SSBJT5QVbIu0ceCIN+18Ie6BUuxMcvD9j4f8B4nwnV3SBWTkCp'
                                                'g69VOF3aXyAgmzxYQ4pw3oJufq2/7IY4WZ15WgReSMVVFmhUVh'
                                                'f+erT61GHs8erEgh8diJAG1qGzlpJIuKm7RQ1m1s88Z1Utyn2+'
                                                'JB4yuRjPcH4tmEsZdZmQ4VAmkcF5iEM+JUl4cRCUjrjpdZhZQC'
                                                'ItPvyOpS3Q9oblsFt/pcYLtxqBVx/qRpfD56ye+ayv8nvPyFCn'
                                                'Gh+sxgA9c2BBuEvJun7Sod1+6GSJ91pLUrbZXj283aZ3kzG5Us'
                                                'VJq5j8/05C/zCs/SZIkVy//ntpoziMg70acoF7AcFJ5rkA8p7f'
                                                'NCCVZl2+h5xgb/gpbq7DnfjxOcyMfi4syFGV0tHZ+LlXYTmDQW'
                                                'tXuMkmEXG49BH4+UDWBQsYGs7tpfUF3MRccHj+JsiGm4LiR0MN'
                                                'daDcMl9OZjEvgy6rTuxcY4ZLKYHHExV/5c8eKon64MVmO8RSYn'
                                                'ykQyQSfqn0zcNGIJMHAmHxChmRjn8D3YPbNqFWdFU5QPjaqAh7'
                                                'E3tHteTLbVBglTaJ1s6NPRz3nvfCPutEbVLCKxGfMPVYENA5+4'
                                                'JMHJUsPjMGJuISdf0dXfiybaahVjeFQbxOM5rJclkuvwH1XdjB'
                                                'szQELjIG90eJhJlj3u'
                                            ),
                                        },
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//R7PHqxIIfHYiQBtahs5aSSLipu0UNZtbPPGdVLcp9'
                                                'vil1mFlAIi0+/I6lLdD2huWwW3+lxgu3GoFXH+pGl8Pnp1pLUr'
                                                'bZXj283aZ3kzG5UsVJq5j8/05C/zCs/SZIkVy2oVZ0VTlA+Nqo'
                                                'CHsTe0e15MttUGCVNonWzo09HPee98a3XdRJQmDyFlO/zYpgzt'
                                                'SSBJT5QVbIu0ceCIN+18Ie6EOKcN6Cbn6tv+yGOFmdeVoEXkjF'
                                                'VRZoVFYX/nq0+tRtdaDcMl9OZjEvgy6rTuxcY4ZLKYHHExV/5c'
                                                '8eKon64MZ+rnrmF3DMUeJDG5xug/LMFhBAzHfdBR00OLvInyv7'
                                                'f4zV+NHi424odMgbRAoi2+qdQMrB3AqDBIqalXT9cQ7VZjvEUm'
                                                'J8pEMkEn6p9M3DRiCTBwJh8QoZkY5/A92D2zgVLsTHLw/Y+H/A'
                                                'eJ8J1d0gVk5AqYOvVThd2l8gIJs8X/57aaM4jIO9GnKBewHBSe'
                                                'a5APKe3zQglWZdvoecYG/yPutEbVLCKxGfMPVYENA5+4JMHJUs'
                                                'PjMGJuISdf0dXfiybaahVjeFQbxOM5rJclkuvwH1XdjBszQELj'
                                                'IG90eJgXG49BH4+UDWBQsYGs7tpfUF3MRccHj+JsiGm4LiR0MN'
                                                'gPE6BfGUWdYssnG53Wgb/SDoiYlT6Q9Ra4BQTByYboRETtzxYK'
                                                'O5E0KosYmmBTnpnOHskiuF2sm9X3QYA9XWncWjBZhuhLj0igwW'
                                                'j40IZJ9MCnJ25dH+5WreyLZjzdS+KEbolMvCfKmneFLC29yXnw'
                                                'Vq8AGfCZEw/PR5VOif0ziy0k0Teu4U2ngr+1QaaQPgk0wf4aXh'
                                                'E8pkbyPamFu9sZsD5Y'
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'invented',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYMAEAAAUAAE=',
                                        },
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYEABjCACAAAZAAI=',
                                        },
                                    ],
                                ],
                                [
                                    'populations',
                                    [
                                        (
                                            '6fba85ebaa5734e567c55e9b2574cc795d717e9939d24d6ceafb5b'
                                            '2d8450db80'
                                        ),
                                        (
                                            'f8261f58da82c8fbc770366d89fae846c3488be512ebf5d47c1644'
                                            'c35906ae14'
                                        ),
                                    ],
                                ],
                                [
                                    'reported',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNrrY2BgiATiWCCuAmIhIHYE4gIgLgNiLiDWAGI9IBYAYmEgjg'
                                                'ZibyDmAOIgIJYE4nggdgJiAPZYBTU='
                                            ),
                                        },
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNoNwyEOgDAQALAThEwuoPaDfWOzGIJigqDwKN5Pm/SLiGq2uP'
                                                'iavBweNldvu5uTp7uPsz/CJASx'
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'stable',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjZMQEAADmABU=',
                                        },
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjZMQEAADmABU=',
                                        },
                                    ],
                                ],
                            ],
                        },
                    },
                    '1e186bb449624fe1f58832593a0149d0b567d60a5752dcab2e97c5283a845e43': {
                        'tests': [
                            (
                                'test_node_list.py::TestAgainstLiveRN::'
                                'test_depth_three[hidden_only-None]'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'max_depth': 3,
                                            'band_threshold': 0.3,
                                            'max_weight': 8.0,
                                            'recurrence': {
                                                'enabled': True,
                                                'allow_hidden_to_hidden': True,
                                                'iteration_level': 2,
                                                'allow_backward': False,
                                                'allow_lateral': False,
                                                'allow_self_loops': False,
                                            },
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            2,
                            'rn',
                            1e-06,
                        ],
                        'value': {
                            'dict': [
                                [
                                    'connections',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNp7zMrA8JKJgWEDMwPDUhYGBjEGBoYXQL4pUJwDiKOBfAUg/g'
                                                '3E7kA8CYgbgPIzgdgRyJ4ApEFAH6j3GiMDAwDDUgnA'
                                            ),
                                        },
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjYGBgaGBiYNjAzMDgDmQrALEYEJuyMjC8AIpfY2RgmACkHY'
                                                'FiM4H0JCAdDcT6LAwMi4D8UKA8kGIQAtJALQwASr4HUA=='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'fitness',
                                    [
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNoT+NNsF7P3mB0Dg4O93/L/QFrBXoaBwT6amcG+yWwKkM9gzw'
                                                'gUY2F4AGQb2IP4ILUQmgHO33TiP1gtAAYgEJ0='
                                            ),
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjYHCwZ4BjA3tGBgUgrWAfzcxgL8PAAGRjx5tO/LeDsA3sZ4'
                                                'nYgcVCnRjsAXVnCsM='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'graphs',
                                    [
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//WYXl7QPL9JHb6RdkA1oQRJECPfBiICDNe7JfOEU6E'
                                                'vZMLH/r8UqhM+cbx3Hx/h5eFT3GyMml9O/YqkauzoW/zXEj2+F'
                                                'Fv/jOXS/pemFE6HeRDbO9kA4GkTkpbFCNujZ9FEJ0uJzJrYT44'
                                                'i6Ph2zPfP/JXXxNebTkSJcjSOdGGDkwy9/kEdlkKVmgqvyMfmi'
                                                'YGcrGOIBYe2zqdcGLZ4ay6vD9w8ZRCnB1o6iNeLwf3stIKGPP2'
                                                'rzTWcy09y4dFBFH7AKQqz5m+LeCbWC5prDUNR0NxZeLpV+f5AT'
                                                'WzfGwGLRUgCLdbIIPY+NkrQSlg2QelxJVQumi7/mhy06iA2WJH'
                                                '50Td+2qqEoTUr5VZlbOi3H1UPMUwEkFGW7jRNTfdSQKhv+DheH'
                                                'tSDPywGzssbb/l9PtcynyZyG+r/v/jJ6FaJAjR/0WBdkdxbSGh'
                                                'LWkvpuxIr2Wc6pTPQf6ZUE5txVcVvC5o66i//nqcBKKmoJGFXt'
                                                'F81gDE+J7jibJm/+nXqc28c//8NIxQJG2Wqx0QEkdszpqTLA0e'
                                                'JhaLI71F+s1hDMR17Cv43iTXwcAR9/dOZmL4vnT9qimPV0aCuD'
                                                'OqiL2R/sgVvQu+ZECDpwTFqb9YmbWFtorPvqQHfdu9D6I1g9+c'
                                                'YEhG9w7Ymwb12KY8a+Mn7ZnztZY1/7o/AllrH6H/L/SNtB/won'
                                                'L3E1XzkhJ6BfvPA3PvDJxN3vPlhjOijllmvjsMRCmPwcFJr79M'
                                                'iZb7kkJ65B5GSbk0yklZkbeFK4VTKJE7W064P6BvFlHaPmSYNC'
                                                'iUqNb6dsn5GRfN4keAuzkNIib9SDtW12e8zUKc0YGYKKDfxAU2'
                                                'MYqF0Dp2xh8MRFHkYU'
                                            ),
                                        },
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//eOwxEKY/BwUmvv0yJlvuSQnrkHkZJuTTKSVmRt4Ur'
                                                'hVR17Cv43iTXwcAR9/dOZmL4vnT9qimPV0aCuDOqiL2R/Ej2+F'
                                                'Fv/jOXS/pemFE6HeRDbO9kA4GkTkpbFCNujZ9MLmjrqL/+epwE'
                                                'oqagkYVe0XzWAMT4nuOJsmb/6depzbG/4OF4e1IM/LAbOyxtv+'
                                                'X0+1zKfJnIb6v+/+MnoVokDDL3+QR2WQpWaCq/Ix+aJgZysY4g'
                                                'Fh7bOp1wYtnhrLq7AKQqz5m+LeCbWC5prDUNR0NxZeLpV+f5AT'
                                                'WzfGwGLRw/cPGUQpwdaOojXi8H97LSChjz9q801nMtPcuHRQRR'
                                                '+Q0iJv1IO1bXZ7zNQpzRgZgooN/EBTYxioXQOnbGHwxEjbQf8K'
                                                'Jy9xNV85ISegX7zwNz7wycTd7z5YYzoo5ZZrxgSEb3DtibBvXY'
                                                'pjxr4yftmfO1ljX/uj8CWWsfof8v/sgVvQu+ZECDpwTFqb9Ymb'
                                                'WFtorPvqQHfdu9D6I1g9+cc//8NIxQJG2Wqx0QEkdszpqTLA0e'
                                                'JhaLI71F+s1hDMdE3ftqqhKE1K+VWZWzotx9VDzFMBJBRlu40T'
                                                'U33UkCoyiRO1tOuD+gbxZR2j5kmDQolKjW+nbJ+RkXzeJHgLs/'
                                                'eOjlNtgckHSFsuJDAxURod0G4HxwIm2VvZizTzv1TIo96mJtqr'
                                                'XDreLFFMaBeaxeSq2PLCrxK9iaZq2+TGa0GtTEfkOMss2EXP2A'
                                                '7FIRQWMpvIbrSvVDBuntQ8Ybu/zZuz9Rb9CElTdR385V5c8yt8'
                                                'rUEYOgI/IHFtpVv4DEPsEqt74TV4T6ZPBP39/WoGPnAo5hCXRd'
                                                'gbatamO/2sWX73M0cG'
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'invented',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYEACjBAKAAAbAAI=',
                                        },
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYGRABYyMjAAALQAF',
                                        },
                                    ],
                                ],
                                [
                                    'populations',
                                    [
                                        (
                                            '6fba85ebaa5734e567c55e9b2574cc795d717e9939d24d6ceafb5b'
                                            '2d8450db80'
                                        ),
                                        (
                                            'f29ad957c49dbe2aa0b61796a88dc2ac1a87255a9a0eb7a433c443'
                                            'd9ecc47cb9'
                                        ),
                                    ],
                                ],
                                [
                                    'reported',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNp7zMrA8JKJgWEDMwPDUhYGBjEGBoYXQL4pUJwDiKOBfAUg/g'
                                                '3E7kA8CYgbgPIzgdgRyJ4ApEFAH6j3GiMDAwDDUgnA'
                                            ),
                                        },
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjYGBgaGBiYNjAzMDgDmQrALEYEJuyMjC8AIpfY2RgmACkHY'
                                                'FiM4H0JCAdDcT6LAwMi4D8UKA8kGIQAtJALQwASr4HUA=='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'stable',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYGBgYARhEAAxgRAAAF8ACg==',
                                        },
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjZGBgZGQEEiDICOYwAgAAhwAN',
                                        },
                                    ],
                                ],
                            ],
                        },
                    },
                    '20bb6f8cccd0354c9da0002ab1e72942f9f5217e799ee587e50626d5da16c446': {
                        'tests': [
                            (
                                'test_node_list.py::TestAgainstLiveRN::'
                                'test_rn_populations[1-with_lateral-2-None]'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'max_depth': 1,
                                            'band_threshold': 0.3,
                                            'max_weight': 8.0,
                                            'recurrence': {
                                                'enabled': True,
                                                'allow_hidden_to_hidden': True,
                                                'iteration_level': 2,
                                                'allow_backward': False,
                                                'allow_lateral': True,
                                                'allow_self_loops': False,
                                            },
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            2,
                            'rn',
                            1e-06,
                        ],
                        'value': {
                            'dict': [
                                [
                                    'connections',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNotyUsKACAMA9GgotiFP+j9r+oUungwJC7pYuGhZcd2MNBRk/'
                                                'KLLnBMbBg+J8gA2g=='
                                            ),
                                        },
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNoTZIAAZihmA2JBKFsUiFmAmBOIBaB8PiBmAmIuIBaB6gXJC0'
                                                'FpAB8EAMs='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'fitness',
                                    [
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNq7bKhgz8zgYM/AwGBvEN5ox8SgYH9Ni8G+IUfJvmYhu70AUH'
                                                'wZr7r9kxAT+w/aBkB1DvZcQDG3pQr2V2QZ7JWWMgHFFOw53O7Y'
                                                'bWZmsAcABT4TXg=='
                                            ),
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjZnCwZwDiJyEm9h+0DeyX8arbN+Qo2bstVbC/bKhgz8SgAJ'
                                                'RXsK9ZyG6vtJTJ/poWg/0VWQb7zcwM9nP+2YP1mmra2DvlO9hX'
                                                'JSrYAwBkBhWz'
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'graphs',
                                    [
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//dRDxF0txIZUAIsNKRzbDoj4FIA2CxJR18e6iqBX5x'
                                                'QwijM31PqYC693a1GLKjpqeAbgdNQ7dMKxJM24Bw2XL81rfBlG'
                                                'LbFg5L2FMI+v/sY90rmkPqkUbo+2KK5q4Kfp5VGYgx9lwnAUjG'
                                                'J/0uwslrgKZ2+EnT0XHBKSDzFvmCtRLcicdY3XbeXzzdLmem3q'
                                                'pcngv92mnQ9l15Qj00ZDC/l/20PCb3Dll9EbUMUd9BRamqor8e'
                                                'TpARVzKEroi/ffZUTPlEyNG49QG6ehcrY2uY/FfIv9idosBQuk'
                                                '1bgjLH3tT1m4+cLth8lY6ndMUn310uJbAi3T+afxwid1srs/+2'
                                                'n1UcNQ7I6sPtEQGOWu8txFhRC72hSoVRBFXRiBseBn+A1xOC38'
                                                '8xMnXeZ/eHrTke/EtTOnKX1UDVumGVSUsCiPetf8bDF6LiSCje'
                                                'jdXkoYIPvuBXhtZSr74KoTHRAa0yZVTSZlBJ8aCBG25h/Dg/d2'
                                                'GqQHwe1YPSYzKVL+/52weuOwxEKY/BwUmvv0yJlvuSQnrkHkZJ'
                                                'uTTKSVmRt4UrhVPTNePO5Qi5UNuDAI00ubj9T80Yl9s9mWPJmi'
                                                'iva/wBjE1d/n/oceGDZFcIjoSzMKzSBO7uGTb3nrYyL+0ryj7I'
                                                'SbyQOLuc1Hbj6pte82LLrY7kwh9m7zo4TrGLbd3I/EH7hEYufz'
                                                '7q1zfRRnKWdEqBWRPXua7HLNuwyZMGlmKsui7t0Zv5vnNlVMJ5'
                                                'f18Ys/90j+bay/SbcswyqjmvdJDFcr8AOEBt+KdqH8BB6SIddC'
                                                '/Q1FRS6N9E3VQ0y9qSv4ltjKJ8BXH2nRxcgeAYVSllbvz3aScu'
                                                '7ZfOtvDZQR3t/TuEcb'
                                            ),
                                        },
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//YozN9T6mAuvd2tRiyo6angG4HTUO3TCsSTNuAcNly'
                                                '/N47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFV61/xs'
                                                'MXouJIKN6N1eShgg++4FeG1lKvvgqhMdEBrTJlVNJmUEnxoIEb'
                                                'bmH8OD93YapAfB7Vg9JjMpUv7/nbB6DXE4LfzzEydd5n94etOR'
                                                '78S1M6cpfVQNW6YZVJSwKI9Ez5RMjRuPUBunoXK2NrmPxXyL/Y'
                                                'naLAULpNW4Iyx97cTV3+f+hx4YNkVwiOhLMwrNIE7u4ZNveetj'
                                                'Iv7SvKPs1EPEXS3EhlQAiw0pHNsOiPgUgDYLElHXx7qKoFfnFD'
                                                'AtyJx1jddt5fPN0uZ6beqlyeC/3aadD2XXlCPTRkML+aLu3Rm/'
                                                'm+c2VUwnl/Xxiz/3SP5trL9JtyzDKqOa90kMT1m4+cLth8lY6n'
                                                'dMUn310uJbAi3T+afxwid1srs/+2kfuERi5/PurXN9FGcpZ0So'
                                                'FZE9e5rscs27DJkwaWYqy3/bQ8JvcOWX0RtQxR30FFqaqivx5O'
                                                'kBFXMoSuiL999lhJvJA4u5zUduPqm17zYsutjuTCH2bvOjhOsY'
                                                'tt3cj8SW2MonwFcfadHFyB4BhVKWVu/PdpJy7tl8628NlBHe31'
                                                '201zNkgRYZiU1Pfa+Zo0EcNqcpui4QNx4k7IvvxjcP47DEQpj8'
                                                'HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFUDqebNpETGhy2R04'
                                                'eRl6FqSq8IWgrQSCwnPDv9jPxJ4MIzvRIRrQ3vu8bQ7CtLDwxd'
                                                '3bUw09PjoLdp1Ews1peAPYI1LpfNBaR8bw0cGavm2sxiB0aJmO'
                                                'EHNptpnbg2KjclfUSE'
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'invented',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYIAARgYEAAAAIAAC',
                                        },
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYEAFjIwMDAAAGwAD',
                                        },
                                    ],
                                ],
                                [
                                    'populations',
                                    [
                                        (
                                            '6fba85ebaa5734e567c55e9b2574cc795d717e9939d24d6ceafb5b'
                                            '2d8450db80'
                                        ),
                                        (
                                            'cb1cfa38706576290397451c3bf9a78b030cb2fc896c8bbc850896'
                                            '4503ab5847'
                                        ),
                                    ],
                                ],
                                [
                                    'reported',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNotyUsKACAMA9GgotiFP+j9r+oUungwJC7pYuGhZcd2MNBRk/'
                                                'KLLnBMbBg+J8gA2g=='
                                            ),
                                        },
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNoTZIAAZihmA2JBKFsUiFmAmBOIBaB8PiBmAmIuIBaB6gXJC0'
                                                'FpAB8EAMs='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'stable',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjZMQEAADmABU=',
                                        },
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjZMQEAADmABU=',
                                        },
                                    ],
                                ],
                            ],
                        },
                    },
                    '245c81906facbd5e4cc86dbc3caed04998cd7cdb5a82ab9f53a5667557062d08': {
                        'tests': [
                            (
                                'test_node_list.py::TestAgainstLiveRN::'
                                'test_rn_populations[2-full_recurrent-3-20]'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'max_depth': 2,
                                            'band_threshold': 0.3,
                                            'max_weight': 8.0,
                                            'recurrence': {
                                                'enabled': True,
                                                'allow_hidden_to_hidden': True,
                                                'iteration_level': 3,
                                                'allow_backward': True,
                                                'allow_lateral': True,
                                                'allow_self_loops': True,
                                                'activate_time': 20,
                                            },
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            2,
                            'rn',
                            1e-06,
                        ],
                        'value': {
                            'dict': [
                                [
                                    'connections',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNp7x8DAcASI5wLxeiCWAOKHQPwdiO8AsTEQOwFxJRDLAPEcIL'
                                                '4J1eMHxB+A2B2q9hoQAwAJFgxF'
                                            ),
                                        },
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNp7x8DAMAeIZYC4EoidgNgdiCWA+DsQH4Hih0D8AYivAfEdIP'
                                                'YD4pNAfBqILwHxEyBOAWIA7gcMvw=='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'fitness',
                                    [
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjYHCwn9rDYC/D2mDHwMBgz8CgYM8DpM8EC9tD+CBsAMc99X'
                                                'ZgsRPCTGCaDaoHphYA9d4LKA=='
                                            ),
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjYHCw76m3s2dgMEDCCmB8JljY/oQwk/3UHgZ7HgYGezYgZk'
                                                'DCt2Qb7UDqWYHsq1GMYD0AKwMNhA=='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'graphs',
                                    [
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//ZzoaqUt2GT2rsCA1qaUsyQvnBcd1/0bj5VJyPxwGL'
                                                'JH41tQqzVp6i8h6kPsswX2GdEboSVupftV3rVwWhSVCOcN/AiV'
                                                'PdL+RdjQb9JFDwVrnA4slHNcGPEsff+Vv3spOe55BXXeLnfhqe'
                                                'lJ/Wnbgt/uTBeH4OZmTYLTGdZZrwPEex+/suS5VgKYRH3UCWp/'
                                                'ORiefT0qgeagP5nq2IXJx0ji4blpPXjdPveF7Qf39HCLksmZIH'
                                                'wmbXSclLe6mApIy4uihfJaJ/uHPaNuioGdsFTgf4pOXvZkSkyS'
                                                '1470gFgBZ46HeFpp9wvu5oHaTHbUmPW4j9DPrsMfoVglQW6ZrQ'
                                                'JU7zD0FGYqXMTMCfbObUvNztahDgw/Qg59xQhjCjzWM5aLhpsV'
                                                'Ax11HAwLwED/K58tGnFzsoP8H51otqbhPDlwSN5McfkSjG3hr2'
                                                'hEYr1jqbXPKX/FXKlfioP9OT4UV5iSfzYfQt7LFwotKYbxXBUn'
                                                'h6w9k6i9BtSdSA1M02ISfx4dc8YIj1nmoL9EUbWkyAOIOm7EFU'
                                                'DNhahMiDh3iuCeSFi/3HJ2LczGER8pDVsOLyWeAJBcQa13Z2sF'
                                                'piM7OG38EpUKkjhwF+Sm+zsh1pWS/GEph5vCQXtLPx8v0I7/ge'
                                                't6vteKvmlyEBQfu+SY07IwZ537M7im/wcnRENQSeTcrOi7xyes'
                                                'Ci0DLckN79PgpTaIz1WWUKY2jx77jBWHr/jQsvuUI6C712V9Pa'
                                                'pPQ6CvviwyiOpGRlrz45XTFpgG9wAih6w/9eMX93rw5v3H3k6v'
                                                'H5RkJelx9MhSB+eKasytjp/JRZ5OoyR+SQaxJoTqDX68/Z6l67'
                                                'Gs9KYQwahu52LcYUD8'
                                            ),
                                        },
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//ZzoaqUt2GT2rsCA1qaUsyQvnBcd1/0bj5VJyPxwGL'
                                                'JHHh1zxgiPWeagv0RRtaTIA4g6bsQVQM2FqEyIOHeK4J6SfzYf'
                                                'Qt7LFwotKYbxXBUnh6w9k6i9BtSdSA1M02ISf0jeTHH5Eoxt4a'
                                                '9oRGK9Y6m1zyl/xVypX4qD/Tk+FFeYlouGmxUDHXUcDAvAQP8r'
                                                'ny0acXOyg/wfnWi2puE8OXDQsvuUI6C712V9PapPQ6CvviwyiO'
                                                'pGRlrz45XTFpgG93sfv7LkuVYCmER91AlqfzkYnn09KoHmoD+Z'
                                                '6tiFycdIi6KF8lon+4c9o26KgZ2wVOB/ik5e9mRKTJLXjvSAWA'
                                                'H8EpUKkjhwF+Sm+zsh1pWS/GEph5vCQXtLPx8v0I7/geNbUKs1'
                                                'aeovIepD7LMF9hnRG6ElbqX7Vd61cFoUlQjn4uG5aT143T73he'
                                                '0H9/Rwi5LJmSB8Jm10nJS3upgKSMus6LvHJ6wKLQMtyQ3v0+Cl'
                                                'NojPVZZQpjaPHvuMFYev+I6fyUWeTqMkfkkGsSaE6g1+vP2epe'
                                                'uxrPSmEMGobudiACKHrD/14xf3evDm/cfeTq8flGQl6XH0yFIH'
                                                '54pqzK3rer7Xir5pchAUH7vkmNOyMGed+zO4pv8HJ0RDUEnk3F'
                                                '82yAI0y5OjPSEooWORsV5tGt/EB0VRm1qPImUTEafpCWfWidi/'
                                                'AiqOxuNfq0IbDccM6FM5PODCBvuePJNtEcfCYSItIaVrjwzAtu'
                                                'u7kl/IhliK37kKOLw41WeTehPywS++TOklrIEeezFYmubCkPM6'
                                                'Ern/Qp9pNy/MWLG0cLMc2IVCAmPJIhhEeuSiNIGFNeA+mtNlbi'
                                                'SllFnkoyIEekrlgT+g'
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'invented',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYMAEAAAUAAE=',
                                        },
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYMAEAAAUAAE=',
                                        },
                                    ],
                                ],
                                [
                                    'populations',
                                    [
                                        (
                                            '6fba85ebaa5734e567c55e9b2574cc795d717e9939d24d6ceafb5b'
                                            '2d8450db80'
                                        ),
                                        (
                                            'c506c5c74656b565ec9eb88362a95205d34b37774a5c04d2b78a6f'
                                            '88866abeb3'
                                        ),
                                    ],
                                ],
                                [
                                    'reported',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNp7x8DAcASI5wLxeiCWAOKHQPwdiO8AsTEQOwFxJRDLAPEcIL'
                                                '4J1eMHxB+A2B2q9hoQAwAJFgxF'
                                            ),
                                        },
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNp7x8DAMAeIZYC4EoidgNgdiCWA+DsQH4Hih0D8AYivAfEdIP'
                                                'YD4pNAfBqILwHxEyBOAWIA7gcMvw=='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'stable',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjZMQEAADmABU=',
                                        },
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjZMQEAADmABU=',
                                        },
                                    ],
                                ],
                            ],
                        },
                    },
                    '3c6dbb685eda5a21d551d8416a8c97a9c0732b4492cb4bca5ba00cb7352f69df': {
                        'tests': [
                            (
                                'test_node_list.py::TestAgainstLiveRN::'
                                'test_invented_connections_are_not_ported'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'max_depth': 2,
                                            'band_threshold': 3.0,
                                            'max_weight': 8.0,
                                            'recurrence': {
                                                'enabled': True,
                                                'allow_hidden_to_hidden': True,
                                                'iteration_level': 2,
                                                'allow_backward': True,
                                                'allow_lateral': True,
                                                'allow_self_loops': True,
                                                'activate_time': 20,
                                            },
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            1,
                            'rn',
                            1e-06,
                        ],
                        'value': {
                            'dict': [
                                [
                                    'connections',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': 'eNpjYYAAXSjNgoRdkMRgtD4SH1kuCEoDACk4ARM=',
                                        },
                                    ],
                                ],
                                [
                                    'fitness',
                                    [
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNork7K3Z2BwsD+jqQCmy4B8GJZqTYeLweiWO95wPrLcxrvSYB'
                                                'oA+KkS1A=='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'graphs',
                                    [
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNq7wS+8IF7SdW7SaXXpudca91/i65gx1W7CV7EdrCwHT7a9eL'
                                                'zhiNOMPzIis35/OTEzf6eK+jrHJymzJ/ssmTpTuiJoR2hzppHe'
                                                'pqdik4y6jCeI88Wo7OwSNbyVZpZTkhrecSFo3itC+m8QsJ9S+a'
                                                'snHr904Cr89aw4SXaaZnWUn8vsPY8er36rsm961ptX65IodR+l'
                                                '+s/H758op6m9qfir9CERhjVnxHyXVZeu/XBi3yq9ePeNBsWU2k'
                                                '9r/63WWvQ5cGJehxzbjjO90c58IqxGm3SWrJJa+8D6h++iVTmE'
                                                '9AMA00NBXw=='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'invented',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjZGRgBAIwAQEMjAAAxQAS',
                                        },
                                    ],
                                ],
                                [
                                    'populations',
                                    [
                                        (
                                            '6fba85ebaa5734e567c55e9b2574cc795d717e9939d24d6ceafb5b'
                                            '2d8450db80'
                                        ),
                                    ],
                                ],
                                [
                                    'reported',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': 'eNpjYYAAXSjNgoRdkMRgtD4SH1kuCEoDACk4ARM=',
                                        },
                                    ],
                                ],
                                [
                                    'stable',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjZAQDBkYkAAAA2QAU',
                                        },
                                    ],
                                ],
                            ],
                        },
                    },
                    '45059ad48d060caff480313533b6b382e27594d8e570e179d65f029dfad9a412': {
                        'tests': [
                            (
                                'test_node_list.py::TestAgainstLiveRN::'
                                'test_rn_populations[2-hidden_only-2-None]'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'max_depth': 2,
                                            'band_threshold': 0.3,
                                            'max_weight': 8.0,
                                            'recurrence': {
                                                'enabled': True,
                                                'allow_hidden_to_hidden': True,
                                                'iteration_level': 2,
                                                'allow_backward': False,
                                                'allow_lateral': False,
                                                'allow_self_loops': False,
                                            },
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            2,
                            'rn',
                            1e-06,
                        ],
                        'value': {
                            'dict': [
                                [
                                    'connections',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNprYGBg8AFifyDOBWI+ILYA4gQonwOIFYFYHYh5gVgAqtYJiN'
                                                'mA2BOIxYA4FIitgRgA12AEjw=='
                                            ),
                                        },
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpTZGBgyAViPiDmBWIxIFYH4gQgZgNifyD2AWILIPYEYgEgtg'
                                                'ZiJyDmZoAAIyCug7IBm+gDsQ=='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'fitness',
                                    [
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjYGCwD2FksJ8DxAxAdruBlT0PkHZNAvEdwGLHFfLt34Qz2v'
                                                'NLqtpzMkDUgTCHMgOUr2Cf1PfNDiQGAP4jDOw='
                                            ),
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNo7rpBvz8DgYN9uYGXPL6kKZCvYvwlntHdNYrDnUGawn8PIYB'
                                                '8CxDwMDPacUMwAxfOaVMF65YFsE8UZdiA2AGh7Ddc='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'graphs',
                                    [
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//RpbnvO2Em5qmRR9OypDiOtB6vJxQMZ0GSIqDCZEMB'
                                                '5eHuxNAS1CPH8piDgMqXbOUuXqkjwWLiZ7n5Z2QKi2OqoFIgIL'
                                                'sLyt0D2+16tSPRXsMEEUACymyWEQ6uOafxO+6O/xWGnUEsd5f4'
                                                'kR/a1bTDSnfwl8hAvW55Zpgc/pZ7Q9rsuZVP+jjlWvQlwDqO7U'
                                                'NcdPXBpl0OFtjKurVcmjgdEJpoH9JZDxXx+7mKa2ZR2NV0IMNp'
                                                '/4SEuS8O0LYIHcbPsfMkpkx3xTeMnFcnqALcAMa0Ez4ZHnOwKu'
                                                'pAHL6RfDeIRCGcnuseI/9mqVxStRZ6jxsMDs8O3a5chfolyVnr'
                                                'iww2en128k1Ojl7/5++RKA9xvEN5Dx5K0ErM89cHp0iIzSUu50'
                                                '/GbIzwONzkeVSs4jrVhnQ6rVDMmYksINLbkwFFeqYdCHJycaUI'
                                                'TiqfKYPVDPiTsd4Vbs0zEi6Ot2yGWhi2o2cBp8Y5ZdilN98s75'
                                                'ARWaMHQcN4Bw2jAHtcyN1RlG3CgWDGcvJ6BLyc8Z2H7SwHFCrz'
                                                'oZmTU8bdz4TkHaEL7faVF+6vTH6JJaPbxNjeagGSfUn/hLLk4t'
                                                'hW60wpvF2bWe3WeB2YwKZdRAl1NsYFPKuQpXThoQbd42pcyV3g'
                                                'AwxsL03JP6rTUrzxGoOgKATToWMdBZfqAN96o31GPePHoJBfaP'
                                                'ORyv455/Xlv0LGVdg3D+TwDezdWht/dDHDCjBAs/hhUmwUCsMV'
                                                'I9j+lEV99laJ8E2DJKnqpNjmpcyS2A95ajgv/9KaZy68/gKRe6'
                                                'ciidt4sLUk+EIUpVd5kQxaVYErhJy8pS8xwpi3ZxAabTaU07GU'
                                                'k8vUIU/pQWNSsqrDqk'
                                            ),
                                        },
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//YzSUu50/GbIzwONzkeVSs4jrVhnQ6rVDMmYksINLb'
                                                'kweIRCGcnuseI/9mqVxStRZ6jxsMDs8O3a5chfolyVnriuy5lU'
                                                '/6OOVa9CXAOo7tQ1x09cGmXQ4W2Mq6tVyaOB0aGLajZwGnxjll'
                                                '2KU33yzvkBFZowdBw3gHDaMAe1zI3VowQLP4YVJsFArDFSPY/p'
                                                'RFffZWifBNgySp6qTY5qXMkUV6ph0IcnJxpQhOKp8pg9UM+JOx'
                                                '3hVuzTMSLo63bIZfsfMkpkx3xTeMnFcnqALcAMa0Ez4ZHnOwKu'
                                                'pAHL6RfDADDGwvTck/qtNSvPEag6AoBNOhYx0Fl+oA33qjfUY9'
                                                '4FIgILsLyt0D2+16tSPRXsMEEUACymyWEQ6uOafxO+6B7sTQEt'
                                                'Qjx/KYg4DKl2zlLl6pI8Fi4me5+WdkCotjqqCaaB/SWQ8V8fu5'
                                                'imtmUdjVdCDDaf+EhLkvDtC2CB3Gw8egkF9o85HK/jnn9eW/Qs'
                                                'ZV2DcP5PAN7N1aG390McMBlG3CgWDGcvJ6BLyc8Z2H7SwHFCrz'
                                                'oZmTU8bdz4TkHaxaVYErhJy8pS8xwpi3ZxAabTaU07GUk8vUIU'
                                                '/pQWNSvF2bWe3WeB2YwKZdRAl1NsYFPKuQpXThoQbd42pcyV3q'
                                                'flZKMxNFYd2AOSvb5foxqfh2gpKzkhWxhYxYjJRTdz47DEQpj8'
                                                'HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFUU9lLa5ARYUxwVnt'
                                                'Xs9BtVdkDqCQ7LRbewIa3rj8lsScPXUHWipvXQCVdEs2REehcQ'
                                                '/ghn1UCX9VjqyIUlxYsw47DEQpj8HBSa+/TImW+5JCeuQeRkm5'
                                                'NMpJWZG3hSuFV24TjA'
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'invented',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYMAEAAAUAAE=',
                                        },
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYEAFjEAEAAAZAAM=',
                                        },
                                    ],
                                ],
                                [
                                    'populations',
                                    [
                                        (
                                            '6fba85ebaa5734e567c55e9b2574cc795d717e9939d24d6ceafb5b'
                                            '2d8450db80'
                                        ),
                                        (
                                            '355218f8c5cdbc86dcbbd593c9717819093d592c356396f0ee3cfe'
                                            '87a72d5704'
                                        ),
                                    ],
                                ],
                                [
                                    'reported',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNprYGBg8AFifyDOBWI+ILYA4gQonwOIFYFYHYh5gVgAqtYJiN'
                                                'mA2BOIxYA4FIitgRgA12AEjw=='
                                            ),
                                        },
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpTZGBgyAViPiDmBWIxIFYH4gQgZgNifyD2AWILIPYEYgEgtg'
                                                'ZiJyDmZoAAIyCug7IBm+gDsQ=='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'stable',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjZMQEAADmABU=',
                                        },
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjZMQEAADmABU=',
                                        },
                                    ],
                                ],
                            ],
                        },
                    },
                    '4900ec920ed1ddbb2cc84161b74aa2c6eec1d5b49e686ae477a8b8fda22334a4': {
                        'tests': ['test_node_list.py::TestAgainstLiveRN::test_neighbour_scale'],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'max_depth': 2,
                                            'band_threshold': 0.3,
                                            'max_weight': 3.0,
                                            'recurrence': {
                                                'enabled': True,
                                                'allow_hidden_to_hidden': True,
                                                'iteration_level': 2,
                                                'allow_backward': True,
                                                'allow_lateral': True,
                                                'allow_self_loops': True,
                                                'activate_time': 20,
                                            },
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            1,
                            'configured',
                            1e-06,
                        ],
                        'value': {
                            'dict': [
                                [
                                    'connections',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNrbz8DAEAjEGUCcDsQsQLwIiHcB8VoglgBiJiD2BGIuIC4A4j'
                                                'wgbgJiViBezAABW4G4G4gBScsHog=='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'fitness',
                                    [
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNo7Ec9k/+ySgf2laGn7XHV2+zIpe/saTRv7qg3K9v+um9l/c2'
                                                'K0P2plZ//rmI09W6SivcTlQ3YmQPmaPaL2PJOF7RUSGOwZGBzs'
                                                'v/z7b6fFwGAPAKcvG/k='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'graphs',
                                    [
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//ehshYRbKdOvlE66gqtBvSwmQI0f9S+1kUP85QL4Br'
                                                '8wc+2sOMgJF3wMI6yGUc/erks/cO8NxZV0kF44KXDE8XmONM+P'
                                                'gqRa1NMYQ0etegp2W0xvdmqcaYmGeF7f3Hf2zDLAfNlIBDVeyH'
                                                'z8yhdk02kFAkux0AYtKzXxc8CoNWFI2A8ToF8ZRZ1iyycbndaB'
                                                'v9IOiJiVPpD1FrgFBMHJhujD8o/6IoL7VCeK/qXrSdfN/7Ttb5'
                                                'sQQgW1NGJxa/52NomhUjZt/N7mXw9Ddwduyq1c7pud+WZUMIId'
                                                'ufr3vsVEqkytI4IOLp9x8W+wB/bxmFs6bwaRQC1JHbqznkSB9N'
                                                'O00m5Fa4kmis6ZvmdtwRZwGLdfcyIUZQXzVDOKWLqD2Ppyoh7V'
                                                'VaZYD8cMjE5dCmyUJCwakVg57hT3B636g98X5Z4FR5yw59P0cZ'
                                                'nCDqKYmW1e3080xHuHme54zYxGfdnQg/2XE1WR70mUoCwoR4KM'
                                                '5olOI0PviNR+ExCir3fGm5bq9fywjr8R1U9jgZwZXbmAT/2lrv'
                                                'gDp6rpYE404n+jWKFYPc4j6Sd1J2h8epJUV/GSyzUC1nKM037V'
                                                'aK38hhEBCnUq/6DkmHnSuel6WD+RldTWv+bU9ZO8hoxil93F00'
                                                'h1JD++XaUL5Kqt+5ioQsGXEee373DuyTCMkyEsH6WqcYKtkNpa'
                                                'xlZ3Z2DaNqc5E4fI0gIDBDxq76Ge85tuLj/jsMRCmPwcFJr79M'
                                                'iZb7kkJ65B5GSbk0yklZkbeFK4VQvDB7kWOM8dgzeiWhki5vI0'
                                                'KelOrHEiPTK4ShVMy/QcsC6sci9sM+0tI0eB1YD02CPp5m34Aq'
                                                '8XDYx1DNe1DpPa4kMG'
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'invented',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYGBgYGSEYAYoBgAAVAAH',
                                        },
                                    ],
                                ],
                                [
                                    'populations',
                                    [
                                        (
                                            '6fba85ebaa5734e567c55e9b2574cc795d717e9939d24d6ceafb5b'
                                            '2d8450db80'
                                        ),
                                    ],
                                ],
                                [
                                    'reported',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNrbz8DAEAjEGUCcDsQsQLwIiHcB8VoglgBiJiD2BGIuIC4A4j'
                                                'wgbgJiViBezAABW4G4G4gBScsHog=='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'stable',
                                    [{'array': '|b1', 'shape': [20], 'zlib64': 'eNpjZMQEAADmABU='}],
                                ],
                            ],
                        },
                    },
                    '497cf110419b366d5ebc27b8c165a932fea4dcc2aa851d893516fc1149442ceb': {
                        'tests': ['test_node_list.py::TestAgainstLiveRN::test_neighbour_scale'],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'max_depth': 2,
                                            'band_threshold': 0.3,
                                            'max_weight': 3.0,
                                            'recurrence': {
                                                'enabled': True,
                                                'allow_hidden_to_hidden': True,
                                                'iteration_level': 2,
                                                'allow_backward': True,
                                                'allow_lateral': True,
                                                'allow_self_loops': True,
                                                'activate_time': 20,
                                            },
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            1,
                            'rn',
                            1e-06,
                        ],
                        'value': {
                            'dict': [
                                [
                                    'connections',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNoTZ2RgUABiMyAWAmIHIFYG4s8MDAwSQNoQiO2BWBoqpwfEBk'
                                                'CsCKUZgFgLiNWBmAuIAbCdA9Y='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'fitness',
                                    [
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjaHhn/sNY2v77n3r7LAYGewYovvDQ2H4DE4LPA6UZ0WgGND'
                                                '4A53oNGg=='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'graphs',
                                    [
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//W8vCGS6YwD8trq8UwvEOcFl+ZdFur1Q6M9oHtH3ME'
                                                'b1W8hEeMibRD9kumMTq9PJNvNGUo5FGFmM/Dnqpzd1leato+Uk'
                                                'hNI4iOn7/YWj2+W797xZUmpvj6jg2nDyDDu4jGCQh5s/GqoRxP'
                                                'aMulBRbsBdvWa0uX4lMRYk16LjWQRPkHCgv8FfK77Wam2AQxfY'
                                                'e+SQag/ygb+tS7nxuGmpjqLae6794wKyhKMJGki479ckP+RCHa'
                                                'nYhsi+YBejGrm15gdfmq5D+xA+QaeqkCteCkUPm4Uqb6ysbRPj'
                                                'KIKyPT0gAvlbnsDDosEGq986s1BuyCS+Q9vfgKL4pwVqsNgrRd'
                                                'zgMDl6QRlfvjkFIOBKH4S6WJfxUngNMsnw6yZJa3u+cmJW6uy8'
                                                '/xV+TWo5qP6SoSstW3jJNAXKGwMq3EMa1cCXqMejArVBUoPorf'
                                                'hbiuxPk19tZx5SNVVHcP1ECNNvEnshjcz4zI04FUp86fW8nUO2'
                                                'vd3vydM6irklqcDH7joceMEf18Nb/WxLxxuQZcikGvhVv4IY4s'
                                                '1ONgQwwgQEM8LTz+//eGNIWvPBC6Xh25qQ/mpxeWf6jNH01lYx'
                                                'tc20V87460y8y+RnAywJOhXcZmVPtGRWLt+aWCTUQVPF8xUll0'
                                                'TAYRA6VbTO+7ac7no8Q16ia9H/i/cLwBND+SAQRcBaXcReM76G'
                                                'XeUTc8yEUY7U+ZcFgUUnV+2A5lHKH2nLka25VfMZy0NR1pVF6h'
                                                'rLRyEn5qrB9DKiv44FfueNKGmHtmoGiWgWpFr5P8Fk1AGYIHL6'
                                                'gIJ/MZ6KZc3TzNlH6G3xEKkvieYZPnxgZB5rr3uVB/qvIARciK'
                                                'HDtRiEQEwjGn6DpEci'
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'invented',
                                    [{'array': '|b1', 'shape': [20], 'zlib64': 'eNpjYMAEAAAUAAE='}],
                                ],
                                [
                                    'populations',
                                    [
                                        (
                                            '6fba85ebaa5734e567c55e9b2574cc795d717e9939d24d6ceafb5b'
                                            '2d8450db80'
                                        ),
                                    ],
                                ],
                                [
                                    'reported',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNoTZ2RgUABiMyAWAmIHIFYG4s8MDAwSQNoQiO2BWBoqpwfEBk'
                                                'CsCKUZgFgLiNWBmAuIAbCdA9Y='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'stable',
                                    [{'array': '|b1', 'shape': [20], 'zlib64': 'eNpjZMQEAADmABU='}],
                                ],
                            ],
                        },
                    },
                    '6c14dfb1e38a30c25446524a67f4f59980d8d0e3f699c28a4efa51d0ffe0cf01': {
                        'tests': [
                            (
                                'test_node_list.py::TestAgainstLiveRN::'
                                'test_rn_populations[2-with_backward-2-None]'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'max_depth': 2,
                                            'band_threshold': 0.3,
                                            'max_weight': 8.0,
                                            'recurrence': {
                                                'enabled': True,
                                                'allow_hidden_to_hidden': True,
                                                'iteration_level': 2,
                                                'allow_backward': True,
                                                'allow_lateral': False,
                                                'allow_self_loops': False,
                                            },
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            2,
                            'rn',
                            1e-06,
                        ],
                        'value': {
                            'dict': [
                                [
                                    'connections',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNrbycDAMAeIq4C4CYhFgHgVEO8A4k1AzAHEZkCcCMQCQNwAxM'
                                                'uBeBEQKwDxMSA2BuI1QLwOiAGNlwll'
                                            ),
                                        },
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNrbycDAYAbEO4DYGIhFgFgAiBcBcSIQrwLiY0C8BojXAbECEC'
                                                '8HYg4gZgXiDCBeAsRHoPoAbJ8IgA=='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'fitness',
                                    [
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNoT4nSyT/n/306GtcEuhqHBjoFBwZ6HgcE+pEHfngFIQ7CBvX'
                                                'Msm33be0n7gNN/7UBik65IguVYwPIK9kxQtQDzDBB1'
                                            ),
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNoT4nSyZ2AwsA9p0AfSCmDc9l7SftIVSXvnWDZ7HgYGexYgZg'
                                                'JiBjS8KxrB3pL83Q5EAwA5mg2k'
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'graphs',
                                    [
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//U4uZZ4ZOymRFQkm2W8Ay3WR78wEdqN9wcC/idF7rv'
                                                '9bqYebBY/78Vp+vhznaZtdDHzLV/ZE5s3R5eTdJnYr8PKivuO0'
                                                'tKLr9riwkKqyQ5jcX7y3LNg95RdpZaLA35RRkaeRIBAko+3vNa'
                                                'nxQ3CnDzOVgQON/wevds36WC+fWQZb1LwP6ltNK9yxmrlFMvZD'
                                                'N1siTSQusfY9LnbfB52QI9IrKXuEvzpCOK1qVlop9lguGam2vX'
                                                'ER/4M1gXeNLQ5wIP/cGd+EBcm3M+1MlrNydtIvben7qpouYIU4'
                                                'uswvFLiN5dC0x1xYEQflfxVNe3ppEolySm5l9+zb5dJ3U9HYOM'
                                                'Oww2en128k1Ojl7/5++RKA9xvEN5Dx5K0ErM89cHp0iBz2RzqM'
                                                'TfxXVPQXA9s/ZpdwBpCMNmAcDYd21NpBy+oWl1ZKzUsLZ91Pyi'
                                                'dOFiqJXwy7+EO8kDigX+m7DItymPR16mXsoia5O5BXkbyYLc14'
                                                'gFHE3RKHg1VBLqqrfYpuI95hh9KxJaou2c4yuPd0kw9uPRUrL6'
                                                'RX9iHyA6utWsGk3Kqg68sKnt3SieXlweXCRThQLI6vttaDswk8'
                                                'PfLRvifFXlaDP65PH2WArFonNyjURfMyD9Dum8BYTpiI0TYj/R'
                                                'XmxC2aACGKswaPDJjKfGbNJOj5E5ezDmwqXJlIWVZDSNo7PTf0'
                                                'y2nZwKkEvRxZHZEe7F3sr0uB+1ObJ+Hg6mir20euIM4/XFs7df'
                                                'qlC1I0BITGYXn30G5uXPi+1Ddgi2RyXFNTEdBj/KlDK1g7LGdt'
                                                '0oU01UJ6UL78k9efedcTTqrtStXTvxCeq/SaIcwib4z0tHskQD'
                                                'SxvF+OEopnKYDS70TF'
                                            ),
                                        },
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//U4uZZ4ZOymRFQkm2W8Ay3WR78wEdqN9wcC/idF7rv'
                                                '9bHPZHOoxN/FdU9BcD2z9ml3AGkIw2YBwNh3bU2kHL6hb/3Bnf'
                                                'hAXJtzPtTJazcnbSL23p+6qaLmCFOLrMLxS4javbR64gzj9cWz'
                                                't1+qULUjQEhMZheffQbm5c+L7UN2CL1LwP6ltNK9yxmrlFMvZD'
                                                'N1siTSQusfY9LnbfB52QI9J16mXsoia5O5BXkbyYLc14gFHE3R'
                                                'KHg1VBLqqrfYpuI8VeVoM/rk8fZYCsWic3KNRF8zIP0O6bwFhO'
                                                'mIjRNiP9l1ZKzUsLZ91PyidOFiqJXwy7+EO8kDigX+m7DItymP'
                                                'QrKXuEvzpCOK1qVlop9lguGam2vXER/4M1gXeNLQ5wIEjaOz03'
                                                '9Mtp2cCpBL0cWR2RHuxd7K9LgftTmyfh4OpoZHJcU1MR0GP8qU'
                                                'MrWDssZ23ShTTVQnpQvvyT15951xNOqu1K1dO/EJ6r9JohzCJv'
                                                'jPS0eyRANLG8X44SimcpgBXmxC2aACGKswaPDJjKfGbNJOj5E5'
                                                'ezDmwqXJlIWVZD3Kqg68sKnt3SieXlweXCRThQLI6vttaDswk8'
                                                'PfLRvieww2en128k1Ojl7/5++RKA9xvEN5Dx5K0ErM89cHp0iK'
                                                'xgovw/EbCCeVwH88HA3a5FOPqv4bi6DD95Kzq7yb2H0acrbCRe'
                                                'bM2A9FPe/esgIijs5K5c0VfMbapARFmYB8V9v7/LRojCtcNN24'
                                                '/kAw5xY8wAi4V34gVigX0Q/u+XGobekQHC9aa/HSjQlligzldn'
                                                'jHMNunUtm2inYg7TdWnwGEfdeNLuLq4zweUxtzj9bWMtc4tomf'
                                                'QQxXp3lv3fOfqklEK4'
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'invented',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYMAEAAAUAAE=',
                                        },
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYMAEAAAUAAE=',
                                        },
                                    ],
                                ],
                                [
                                    'populations',
                                    [
                                        (
                                            '6fba85ebaa5734e567c55e9b2574cc795d717e9939d24d6ceafb5b'
                                            '2d8450db80'
                                        ),
                                        (
                                            '0dbc2fc6dd354a64636de88a1c4334670d6a93dd6b97799479d1ab'
                                            '99776c966f'
                                        ),
                                    ],
                                ],
                                [
                                    'reported',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNrbycDAMAeIq4C4CYhFgHgVEO8A4k1AzAHEZkCcCMQCQNwAxM'
                                                'uBeBEQKwDxMSA2BuI1QLwOiAGNlwll'
                                            ),
                                        },
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNrbycDAYAbEO4DYGIhFgFgAiBcBcSIQrwLiY0C8BojXAbECEC'
                                                '8HYg4gZgXiDCBeAsRHoPoAbJ8IgA=='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'stable',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjZMQEAADmABU=',
                                        },
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjZMQEAADmABU=',
                                        },
                                    ],
                                ],
                            ],
                        },
                    },
                    '71a9254d1acf88c61ad5f72af6a4e5cc2c415223631dfddf91a3ee94c433d141': {
                        'tests': [
                            (
                                'test_node_list.py::TestAgainstLiveRN::'
                                'test_rn_populations[2-full_recurrent-2-20]'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'max_depth': 2,
                                            'band_threshold': 0.3,
                                            'max_weight': 8.0,
                                            'recurrence': {
                                                'enabled': True,
                                                'allow_hidden_to_hidden': True,
                                                'iteration_level': 2,
                                                'allow_backward': True,
                                                'allow_lateral': True,
                                                'allow_self_loops': True,
                                                'activate_time': 20,
                                            },
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            2,
                            'rn',
                            1e-06,
                        ],
                        'value': {
                            'dict': [
                                [
                                    'connections',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNp7x8DAcASI5wLxeiCWAOKHQPwdiO8AMRcQOwFxJRDLAPEcIL'
                                                '4J1WMGxB+A2B2q9hoQAwD/dwwE'
                                            ),
                                        },
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNp7x8DAMAeIZYC4EoidgNgdiCWA+DsQH4Hih0D8AYivAfEdID'
                                                'YD4pNAfBqILwHxEyBOAWIA68cMpw=='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'fitness',
                                    [
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjYHCwn9rDYC/D2mDHwMBgz8CgYM8DpM8EC9tD+CBsAMc99X'
                                                'ZgsRPCTGCaDaoHphYA9d4LKA=='
                                            ),
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjYHCw76m3s2dgMEDCCmB8JljY/oQwk/3UHgZ7HgYGezYgZk'
                                                'DCt2Qb7UDqWYHsq1GMYD0AKwMNhA=='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'graphs',
                                    [
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//ZzoaqUt2GT2rsCA1qaUsyQvnBcd1/0bj5VJyPxwGL'
                                                'JH41tQqzVp6i8h6kPsswX2GdEboSVupftV3rVwWhSVCOcN/AiV'
                                                'PdL+RdjQb9JFDwVrnA4slHNcGPEsff+Vv3spOe55BXXeLnfhqe'
                                                'lJ/Wnbgt/uTBeH4OZmTYLTGdZZrwPEex+/suS5VgKYRH3UCWp/'
                                                'ORiefT0qgeagP5nq2IXJx0ji4blpPXjdPveF7Qf39HCLksmZIH'
                                                'wmbXSclLe6mApIy4uihfJaJ/uHPaNuioGdsFTgf4pOXvZkSkyS'
                                                '1470gFgBZ46HeFpp9wvu5oHaTHbUmPW4j9DPrsMfoVglQW6ZrQ'
                                                'JB4yuRjPcH4tmEsZdZmQ4VAmkcF5iEM+JUl4cRCUjrjpaLhpsV'
                                                'Ax11HAwLwED/K58tGnFzsoP8H51otqbhPDlwSN5McfkSjG3hr2'
                                                'hEYr1jqbXPKX/FXKlfioP9OT4UV5iSfzYfQt7LFwotKYbxXBUn'
                                                'h6w9k6i9BtSdSA1M02ISfx4dc8YIj1nmoL9EUbWkyAOIOm7EFU'
                                                'DNhahMiDh3iuCeSFi/3HJ2LczGER8pDVsOLyWeAJBcQa13Z2sF'
                                                'piM7OG38EpUKkjhwF+Sm+zsh1pWS/GEph5vCQXtLPx8v0I7/gZ'
                                                'JjUPiLog3fqomj2ktXhQk1WUSjPhtBCFueeMr7t23/rOi7xyes'
                                                'Ci0DLckN79PgpTaIz1WWUKY2jx77jBWHr/jQsvuUI6C712V9Pa'
                                                'pPQ6CvviwyiOpGRlrz45XTFpgG9wAih6w/9eMX93rw5v3H3k6v'
                                                'H5RkJelx9MhSB+eKasytjp/JRZ5OoyR+SQaxJoTqDX68/Z6l67'
                                                'Gs9KYQwahu52L4ZT+o'
                                            ),
                                        },
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//ZzoaqUt2GT2rsCA1qaUsyQvnBcd1/0bj5VJyPxwGL'
                                                'JHHh1zxgiPWeagv0RRtaTIA4g6bsQVQM2FqEyIOHeK4J6SfzYf'
                                                'Qt7LFwotKYbxXBUnh6w9k6i9BtSdSA1M02ISf0jeTHH5Eoxt4a'
                                                '9oRGK9Y6m1zyl/xVypX4qD/Tk+FFeYlouGmxUDHXUcDAvAQP8r'
                                                'ny0acXOyg/wfnWi2puE8OXDQsvuUI6C712V9PapPQ6CvviwyiO'
                                                'pGRlrz45XTFpgG93sfv7LkuVYCmER91AlqfzkYnn09KoHmoD+Z'
                                                '6tiFycdIi6KF8lon+4c9o26KgZ2wVOB/ik5e9mRKTJLXjvSAWA'
                                                'H8EpUKkjhwF+Sm+zsh1pWS/GEph5vCQXtLPx8v0I7/geNbUKs1'
                                                'aeovIepD7LMF9hnRG6ElbqX7Vd61cFoUlQjn4uG5aT143T73he'
                                                '0H9/Rwi5LJmSB8Jm10nJS3upgKSMus6LvHJ6wKLQMtyQ3v0+Cl'
                                                'NojPVZZQpjaPHvuMFYev+I6fyUWeTqMkfkkGsSaE6g1+vP2epe'
                                                'uxrPSmEMGobudiACKHrD/14xf3evDm/cfeTq8flGQl6XH0yFIH'
                                                '54pqzK2SY1D4i6IN36qJo9pLV4UJNVlEoz4bQQhbnnjK+7dt/1'
                                                '82yAI0y5OjPSEooWORsV5tGt/EB0VRm1qPImUTEafpCWfWidi/'
                                                'AiqOxuNfq0IbDccM6FM5PODCBvuePJNtEcfCYSItIaVrjwzAtu'
                                                'u7kl/IhliK37kKOLw41WeTehPywS++TOklrIEeezFYmubCkPM6'
                                                'Ern/Qp9pNy/MWLG0cLMc2IVCAmPJIhhEeuSiNIGFNeA+mtNlbi'
                                                'SllFnkoyIEekr4hD5d'
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'invented',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYMAEAAAUAAE=',
                                        },
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYMAEAAAUAAE=',
                                        },
                                    ],
                                ],
                                [
                                    'populations',
                                    [
                                        (
                                            '6fba85ebaa5734e567c55e9b2574cc795d717e9939d24d6ceafb5b'
                                            '2d8450db80'
                                        ),
                                        (
                                            'c506c5c74656b565ec9eb88362a95205d34b37774a5c04d2b78a6f'
                                            '88866abeb3'
                                        ),
                                    ],
                                ],
                                [
                                    'reported',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNp7x8DAcASI5wLxeiCWAOKHQPwdiO8AMRcQOwFxJRDLAPEcIL'
                                                '4J1WMGxB+A2B2q9hoQAwD/dwwE'
                                            ),
                                        },
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNp7x8DAMAeIZYC4EoidgNgdiCWA+DsQH4Hih0D8AYivAfEdID'
                                                'YD4pNAfBqILwHxEyBOAWIA68cMpw=='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'stable',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjZMQEAADmABU=',
                                        },
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjZMQEAADmABU=',
                                        },
                                    ],
                                ],
                            ],
                        },
                    },
                    '75590d818b916ab21199bb857c0729470700b42e71fa8f60d45101024d4296fb': {
                        'tests': [
                            (
                                'test_node_list.py::TestAgainstLiveRN::'
                                'test_depth_three[full_recurrent-20]'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'max_depth': 3,
                                            'band_threshold': 0.3,
                                            'max_weight': 8.0,
                                            'recurrence': {
                                                'enabled': True,
                                                'allow_hidden_to_hidden': True,
                                                'iteration_level': 2,
                                                'allow_backward': True,
                                                'allow_lateral': True,
                                                'allow_self_loops': True,
                                                'activate_time': 20,
                                            },
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            2,
                            'rn',
                            1e-06,
                        ],
                        'value': {
                            'dict': [
                                [
                                    'connections',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNp7xcXA0MzBwODAzsDwmY2BQZOBgWEpUCyAB8jgZmBYCaRsgF'
                                                'iZlYHhFZDmAqr9BsQvgfg8EwNDFlANCBQB9fRwMjAAALjCCk4='
                                            ),
                                        },
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjYGBg+MbBwPAKSDcDaWVWBoYAHgYGGyBfE4gZuBkYXgLFuY'
                                                'B4KRcDw2c2BoYeTgaGIiCbBSg9EyjuBGQcYWZg2ABkAwB/xQmO'
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'fitness',
                                    [
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjYGCwX/hf377p1W87NiCbgUHB/osXg/3uIEV7rxfyQD5ErP'
                                                '+9PpA2sK8NBPEd7J/nMULlIHwYGwBaJRGm'
                                            ),
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjYHCwZwBjA/uF//Xt+9/r2+8OUgTyFcDY64W8/fM8RvvaQA'
                                                'b7L14M9mwMDEBxCC6Tsoeqg/BvijPYAwAEwxEf'
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'graphs',
                                    [
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//XqUZ4Rxoinj2oCsQ3spXyc+d8u12IMlllWr2o/lIJ'
                                                'GPTbP+tEx/Rx1sJCxOnUcY++hzFbSIHgq+mI/JXvp7O/F6HNSx'
                                                'tkaMAKsJmU/SyA+RtlDj+ccnQW4eLWmXnC0M2uMRlgHThGNzeU'
                                                'EE9azZYXCMIDD0XMXi+ulYKyCQFmXMFzoO1EplkeqXMiIDzEsF'
                                                'V5TN9l+iV05VGzTHpVt/Q/uu3lAadOO2ZUZrTWFcQiySvBdCTW'
                                                'lC24yxdaa+b5FWyATRqdJ4Ji2QDMDCIkaAtj+oJk8weIFmSSjk'
                                                'PNZNuIqdjGTN/BwIc3zefGUVvIbOFkIZS9vONU3tU/cicq+MQy'
                                                'odsI88qWOxaU0bZYw6QEy6iaYhYodYgAst/XkdU1duEOzvdRDy'
                                                'zGxuZVzeLDECqQI9TlOFbGBi3j0QPS7rNxsNwdypZOIHC1a7Qf'
                                                'NkphoHa0o6e0zAi7oadT+ZTBycxnhvcfUaijXhoUqEX8Z8OwLe'
                                                'V4gwaToVYGan31mzuz0ao3+eElGao27PGcWhyVNYjLBUP3YlOK'
                                                'yWonRSNwoZF3ctPvwr26PHyS6jqLa52pLiNj4/iJzw/ZYtrKTq'
                                                'RftWcL+s1i3B75CghT3ukjTrCvN4jPKs9AonVrxFC3/TIVUKL9'
                                                'ET01hH/nbmyjGDx0P7usFlWPE5QdOP6+rRUpwQT9bsUzTxUN9S'
                                                'HbSV350iHr0pnuPyWz9cV2qrc+AhLYRw9mnjsMRCmPwcFJr79M'
                                                'iZb7kkJ65B5GSbk0yklZkbeFK4VQchBbfnnFYbceVJ4C3KY7H6'
                                                '/Gj6sv0ZQyG3f7pTRjucx1sOhZX/WVt6xi4QTDTXWsVQWfXe4V'
                                                'oX8pG6Yzgxy21U6TSo'
                                            ),
                                        },
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//eOwxEKY/BwUmvv0yJlvuSQnrkHkZJuTTKSVmRt4Ur'
                                                'hVPvwr26PHyS6jqLa52pLiNj4/iJzw/ZYtrKTqRftWcL9vcfUa'
                                                'ijXhoUqEX8Z8OwLeV4gwaToVYGan31mzuz0ao02z/rRMf0cdbC'
                                                'QsTp1HGPvocxW0iB4KvpiPyV76ezvxwdypZOIHC1a7QfNkphoH'
                                                'a0o6e0zAi7oadT+ZTBycxngE0anSeCYtkAzAwiJGgLY/qCZPMH'
                                                'iBZkko5DzWTbiKnezvdRDyzGxuZVzeLDECqQI9TlOFbGBi3j0Q'
                                                'PS7rNxsNFzoO1EplkeqXMiIDzEsFV5TN9l+iV05VGzTHpVt/Q/'
                                                'uMZM38HAhzfN58ZRW8hs4WQhlL2841Te1T9yJyr4xDKqzWLcHv'
                                                'kKCFPe6SNOsK83iM8qz0CidWvEULf9MhVQovf54SUZqjbs8Zxa'
                                                'HJU1iMsFQ/diU4rJaidFI3ChkXdy2u3lAadOO2ZUZrTWFcQiyS'
                                                'vBdCTWlC24yxdaa+b5FWyOMRlgHThGNzeUEE9azZYXCMIDD0XM'
                                                'Xi+ulYKyCQFmXMx1sOhZX/WVt6xi4QTDTXWsVQWfXe4VoX8pG6'
                                                'Yzgxy20HIQW355xWG3HlSeAtymOx+vxo+rL9GUMht3+6U0Y7nN'
                                                'gPE6BfGUWdYssnG53Wgb/SDoiYlT6Q9Ra4BQTByYbohks5PL/x'
                                                'pNjFzEjzIs/M92pDme5aB9auYl0SfGifUUR52VlHhzF49uh9Cq'
                                                'jgjzR8Bo/lnFOxZosGz9+CGpt6PZS3d76ihsw1Si+9DuhPB4/p'
                                                'O2e2zSsUlajbAKSEDeByBleK3STLa3P2/Eb+zPQrRWctXPW9eQ'
                                                '+HK4ZrN52neSBqQDWq'
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'invented',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYEACjBAKAAAbAAI=',
                                        },
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYGRABowgBAAALgAE',
                                        },
                                    ],
                                ],
                                [
                                    'populations',
                                    [
                                        (
                                            '6fba85ebaa5734e567c55e9b2574cc795d717e9939d24d6ceafb5b'
                                            '2d8450db80'
                                        ),
                                        (
                                            '178a294edb0a40640c8b715bc1e723166ec6cf39b51450a723fc7e'
                                            '4feea65374'
                                        ),
                                    ],
                                ],
                                [
                                    'reported',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNp7xcXA0MzBwODAzsDwmY2BQZOBgWEpUCyAB8jgZmBYCaRsgF'
                                                'iZlYHhFZDmAqr9BsQvgfg8EwNDFlANCBQB9fRwMjAAALjCCk4='
                                            ),
                                        },
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjYGBg+MbBwPAKSDcDaWVWBoYAHgYGGyBfE4gZuBkYXgLFuY'
                                                'B4KRcDw2c2BoYeTgaGIiCbBSg9EyjuBGQcYWZg2ABkAwB/xQmO'
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'stable',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYGBgYARhRjAF5gAAAEcABg==',
                                        },
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjZGBkYGBgBBEgAKYBAFoABg==',
                                        },
                                    ],
                                ],
                            ],
                        },
                    },
                    '7d283bbf0f2b2ee684a81ea4ee090e94889a96cf25614d8043ecd047ce6e7c43': {
                        'tests': [
                            (
                                'test_node_list.py::TestAgainstLiveRN::'
                                'test_rn_populations[2-hidden_only-0-None]'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'max_depth': 2,
                                            'band_threshold': 0.3,
                                            'max_weight': 8.0,
                                            'recurrence': {
                                                'enabled': True,
                                                'allow_hidden_to_hidden': True,
                                                'iteration_level': 0,
                                                'allow_backward': False,
                                                'allow_lateral': False,
                                                'allow_self_loops': False,
                                            },
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            2,
                            'rn',
                            1e-06,
                        ],
                        'value': {
                            'dict': [
                                [
                                    'connections',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNrTZWBgkANiYSDWA2IOIGYFYgkoHwSEgFgEiDmhciA14kDMBM'
                                                'QqQMwHxPJQNgBKWAGq'
                                            ),
                                        },
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjYGBg0ANiViAWAmJOIOYDYg4gFgZicSBmgspLQMXlgFgFiN'
                                                'UYIAAkJg/EkkAMADR4AV4='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'fitness',
                                    [
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjYGCwZwbiBlYhewYgzcCgYM/CYGAvDWT/+W8P5DsAsYG9QP'
                                                'd3uyf/9e3VgOICQLxoMbM9hzKDPRNUD0Qvgz0AOUUODg=='
                                            ),
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjYHCw//Pf3p6FwcCeAYif/NcH0gpg3MAqZL9oMbM9hzKDvR'
                                                'oDg700EAsAMTMQMwGxirU5UJ2D/dv/DECawf5bF789AMBvEM4='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'graphs',
                                    [
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//b4iBli6mBW5y2wajEJLaCfj5Nzbfnt9zMsbKeZfU9'
                                                'QwJzJopyzjFPmJ4YSqbhUTNt4AYqfDhlaEabXoIDgyjUbNdLO5'
                                                'Qy68KoJVhpIi0a4oPiuZ+8L4sz1tSjrBERnfGLWRy2z6br1SRG'
                                                'sT+9p0KH51xQwPF/HFrffBvl8GJCqwPr9qx/B0XuPiWChx887i'
                                                'eqHseVBzMYtqZW0s6wrxpiasBlm4bYiXUPm0NYHPSC4a8ioKtl'
                                                'XAm1GM818XC1uDgNxnMq3y3RusagmWRZHDFh3uG2nwp2GjgoFG'
                                                'offGpwrIjOU17ogBvJqmZfkeKitmdd9DK6Rff3Ioe2mtO+cJ8a'
                                                'vjsMRCmPwcFJr79MiZb7kkJ65B5GSbk0yklZkbeFK4VdBmQA5y'
                                                'eBdMwGWHv78gJM2mfVBc6duP4JTb9wECq1aFU1Uyp8vcwahhn0'
                                                'TsCtI1sT8gc9X86X5mbpq40/KB5ZJh3oeUuEDG4QUcCAH3/LWZ'
                                                'Nmaci/Dc0NAJgYE+1Kpw2J+iqDNPIZzIztx79R5XYYFQ11g8Cr'
                                                'fq7iacoy3qK97VIs8lhwq5p4Lz2jCL4rSFpVsb64eAipTHsBH4'
                                                '8o3nhHHBeAq4DwBO95JPMFskTggt3sn7IHE1+408Y7+F71Tbxo'
                                                'lk80OEh6VbDk3mWU4YVHlAMI3KPwP4VGBnN3I5ydgSBazc3Vw4'
                                                'BZD48xDjPfFr1e93XK4F8sV5rGhX8NKdfqQds+PY31/WXH6sdG'
                                                'vhTWra+2kxpFO5Wr9sP3+pY2f3jHDlE7X3m5UmVp12YePialWR'
                                                'yl8Fwt0qO07E6sq/BOiM6gXkCRLFQad9cQQmw6Mj//Ud9SibTZ'
                                                'UFmLs3tnhuuTC7E0gD'
                                            ),
                                        },
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//eOwxEKY/BwUmvv0yJlvuSQnrkHkZJuTTKSVmRt4Ur'
                                                'hVjOU17ogBvJqmZfkeKitmdd9DK6Rff3Ioe2mtO+cJ8ausBlm4'
                                                'bYiXUPm0NYHPSC4a8ioKtlXAm1GM818XC1uDgNBmQA5yeBdMwG'
                                                'WHv78gJM2mfVBc6duP4JTb9wECq1aFYd6HlLhAxuEFHAgB9/y1'
                                                'mTZmnIvw3NDQCYGBPtSqcNgds+PY31/WXH6sdGvhTWra+2kxpF'
                                                'O5Wr9sP3+pY2f3jD6/asfwdF7j4lgocfPO4nqh7HlQczGLamVt'
                                                'LOsK8aYmzXSzuUMuvCqCVYaSItGuKD4rmfvC+LM9bUo6wREZ3x'
                                                'jBeAq4DwBO95JPMFskTggt3sn7IHE1+408Y7+F71Tbxolk80OE'
                                                'h6VbDk3mWU4YVHlAMI3KPwP4VGBnN3I5ydgSn6KoM08hnMjO3H'
                                                'v1HldhgVDXWDwKt+ruJpyjLeor3tXcZzKt8t0brGoJlkWRwxYd'
                                                '7htp8Kdho4KBRqH3xqcKyCLPJYcKuaeC89owi+K0haVbG+uHgI'
                                                'qUx7AR+PKN54RxJzJopyzjFPmJ4YSqbhUTNt4AYqfDhlaEabXo'
                                                'IDgyjUYFrNzdXDgFkPjzEOM98WvV73dcrgXyxXmsaFfw0p1+pI'
                                                'oVYXF7PFItB8CssUW3PneWGFo76dQR4XpJ44cslnI847DEQpj8'
                                                'HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFVNFW6J7/Opw7mzdr'
                                                'HCirIOtyKNosgmnYPr5jDTob2Zr6OzkWDRfm/GUhdU9w+Is4ZG'
                                                'nkQr63B8ow7tRxds2VekKHlndsj2ujV8lm+XW973GlvEhloHCL'
                                                'H+3BGPJoKXr6iNSUcy'
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'invented',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYMAEAAAUAAE=',
                                        },
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYEAFjEAMAAAYAAI=',
                                        },
                                    ],
                                ],
                                [
                                    'populations',
                                    [
                                        (
                                            '6fba85ebaa5734e567c55e9b2574cc795d717e9939d24d6ceafb5b'
                                            '2d8450db80'
                                        ),
                                        (
                                            'e6425b661e41a99af773a6e895e4dff7e134d103edb0f5bd86521f'
                                            '446acc10e8'
                                        ),
                                    ],
                                ],
                                [
                                    'reported',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNrTZWBgkANiYSDWA2IOIGYFYgkoHwSEgFgEiDmhciA14kDMBM'
                                                'QqQMwHxPJQNgBKWAGq'
                                            ),
                                        },
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjYGBg0ANiViAWAmJOIOYDYg4gFgZicSBmgspLQMXlgFgFiN'
                                                'UYIAAkJg/EkkAMADR4AV4='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'stable',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjZMQEAADmABU=',
                                        },
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjZMQEAADmABU=',
                                        },
                                    ],
                                ],
                            ],
                        },
                    },
                    '8852f167c904338f35d94733b3edb55212d4d20222d21722a9976a53f366b989': {
                        'tests': [
                            (
                                'test_node_list.py::TestAgainstLiveRN::'
                                'test_rn_populations[2-full_recurrent-1-20]'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [
                                                [-1.0, -1.0],
                                                [0.0, -1.0],
                                                [1.0, -1.0],
                                            ],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'max_depth': 2,
                                            'band_threshold': 0.3,
                                            'max_weight': 8.0,
                                            'recurrence': {
                                                'enabled': True,
                                                'allow_hidden_to_hidden': True,
                                                'iteration_level': 1,
                                                'allow_backward': True,
                                                'allow_lateral': True,
                                                'allow_self_loops': True,
                                                'activate_time': 20,
                                            },
                                        },
                                    },
                                },
                            },
                            {
                                'problem': 'XORProblem',
                                'get_data': [
                                    [[0.0, 0.0, 1.0], [0.0]],
                                    [[0.0, 1.0, 1.0], [1.0]],
                                    [[1.0, 0.0, 1.0], [1.0]],
                                    [[1.0, 1.0, 1.0], [0.0]],
                                ],
                                'fitness_threshold': 0.95,
                                'input_shape': [3],
                                'inputs': [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ],
                                'jitable': True,
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                                'use_bias': True,
                            },
                            42,
                            2,
                            'rn',
                            1e-06,
                        ],
                        'value': {
                            'dict': [
                                [
                                    'connections',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNp7x8DAcASI5wLxeiCWAOKHQHwYiO8wQIATEEcDsQwQ9wFxEx'
                                                'AvBGI+IP4AxAZQtaeAGADalwrV'
                                            ),
                                        },
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjYGBgeAfEfUC8EIhlgDgaiJ2A2ACIJYD4CBDzAfFDIP4AxK'
                                                'eA+A4QvwbiN0DMBMTcQLwUiAGDGwnx'
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'fitness',
                                    [
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjYHCwn9rDYC/D2mDHwMBgz8CgYM8DpMNmzoXyHYDYAI7viN'
                                                'jZg8TNHljaby1isGeD6oGoZbAHADCoDT8='
                                            ),
                                        },
                                        {
                                            'array': '<f4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjYHCwZwDiOyJ29mYPLIFsAySsAMZTexjstxYx2PMwMNizAT'
                                                'EDHDvYd5VY2pto2oD5H5ge2AEAQ1wO9Q=='
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'graphs',
                                    [
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//ZzoaqUt2GT2rsCA1qaUsyQvnBcd1/0bj5VJyPxwGL'
                                                'JH41tQqzVp6i8h6kPsswX2GdEboSVupftV3rVwWhSVCOcN/AiV'
                                                'PdL+RdjQb9JFDwVrnA4slHNcGPEsff+Vv3spOe55BXXeLnfhqe'
                                                'lJ/Wnbgt/uTBeH4OZmTYLTGdZZrwPEex+/suS5VgKYRH3UCWp/'
                                                'ORiefT0qgeagP5nq2IXJx0ji4blpPXjdPveF7Qf39HCLksmZIH'
                                                'wmbXSclLe6mApIyzXaupz89bgVkto3FgND7+obyrW/w2K2/A3R'
                                                'tvienMlkZ46HeFpp9wvu5oHaTHbUmPW4j9DPrsMfoVglQW6ZrQ'
                                                'LjsMRCmPwcFJr79MiZb7kkJ65B5GSbk0yklZkbeFK4VZaLhpsV'
                                                'Ax11HAwLwED/K58tGnFzsoP8H51otqbhPDlwrlzhxmAM7dujeU'
                                                'pIp42JYuBBE3zafBVATOjFuKPmswqSfzYfQt7LFwotKYbxXBUn'
                                                'h6w9k6i9BtSdSA1M02ISfwD3uJToHhH5YCSQocFd/RkhL44Rdx'
                                                'OPjm54WTYjsieY+UCXVPO/dk7YhJLI4EvY0LqPlWih+15otg+C'
                                                'qq6yJ3EXHOjOhzJ+K6AWbi8IK3MSFo9xBH7vnMmBa9Ls0oH9Wk'
                                                'Ui/GarCJdEcVpTRkKxodBxsO6l20dq2ON1C+DsD+UErOi7xyes'
                                                'Ci0DLckN79PgpTaIz1WWUKY2jx77jBWHr/gyIwhXrYtX5SmLp4'
                                                'LG1ZPVe8Z7tMvBgGl92yWeDZJbdwAih6w/9eMX93rw5v3H3k6v'
                                                'H5RkJelx9MhSB+eKasytlmtNiU7iF5ox2PB4fba3PS+gUzpwKM'
                                                'vzc+VphKS714PjIUm6'
                                            ),
                                        },
                                        {
                                            'array': '|u1',
                                            'shape': [20, 32],
                                            'zlib64': (
                                                'eNoBgAJ//eOwxEKY/BwUmvv0yJlvuSQnrkHkZJuTTKSVmRt4Ur'
                                                'hVnOhqpS3YZPauwIDWppSzJC+cFx3X/RuPlUnI/HAYskcA97iU'
                                                '6B4R+WAkkKHBXf0ZIS+OEXcTj45ueFk2I7InmBcc6M6HMn4roB'
                                                'ZuLwgrcxIWj3EEfu+cyYFr0uzSgf1akn82H0LeyxcKLSmG8VwV'
                                                'J4esPZOovQbUnUgNTNNiEn+uXOHGYAzt26N5SkinjYli4EETfN'
                                                'p8FUBM6MW4o+azCpaLhpsVAx11HAwLwED/K58tGnFzsoP8H51o'
                                                'tqbhPDlwMiMIV62LV+Upi6eCxtWT1XvGe7TLwYBpfdslng2SW3'
                                                'd7H7+y5LlWAphEfdQJan85GJ59PSqB5qA/merYhcnHSONbUKs1'
                                                'aeovIepD7LMF9hnRG6ElbqX7Vd61cFoUlQjnRSL8ZqsIl0RxWl'
                                                'NGQrGh0HGw7qXbR2rY43UL4OwP5QTi4blpPXjdPveF7Qf39HCL'
                                                'ksmZIHwmbXSclLe6mApIy6zou8cnrAotAy3JDe/T4KU2iM9Vll'
                                                'CmNo8e+4wVh6/4lmtNiU7iF5ox2PB4fba3PS+gUzpwKMvzc+Vp'
                                                'hKS714MAIoesP/XjF/d68Ob9x95Orx+UZCXpcfTIUgfnimrMre'
                                                'BrunZ8DUbKH3c2RP9EE9u2B3MvOW/RutCNdLqDR0+5A3lgAq/a'
                                                '9orhvX/ksxmpvSrbj8yca3jYoo5Q4e6Ve90wbWEekybjIiX+88'
                                                'VY96g6Jvy3rcF/l8WPXGjIgmnjtZgkz90WeNgWqVpmkbk6nGeT'
                                                'fdEBAygkS4kEgtmtzw3ckagGSKF1IrpoCl9gLMQXHUtJrGscHL'
                                                'IBrnEPkOfdqCILoECw'
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'invented',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYMAEAAAUAAE=',
                                        },
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjYEADjIwMAAAZAAM=',
                                        },
                                    ],
                                ],
                                [
                                    'populations',
                                    [
                                        (
                                            '6fba85ebaa5734e567c55e9b2574cc795d717e9939d24d6ceafb5b'
                                            '2d8450db80'
                                        ),
                                        (
                                            '2e3ca6c227eb420048c3c051f99c70bd12ffeaf930492553082260'
                                            '10149c02db'
                                        ),
                                    ],
                                ],
                                [
                                    'reported',
                                    [
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNp7x8DAcASI5wLxeiCWAOKHQHwYiO8wQIATEEcDsQwQ9wFxEx'
                                                'AvBGI+IP4AxAZQtaeAGADalwrV'
                                            ),
                                        },
                                        {
                                            'array': '<i4',
                                            'shape': [20],
                                            'zlib64': (
                                                'eNpjYGBgeAfEfUC8EIhlgDgaiJ2A2ACIJYD4CBDzAfFDIP4AxK'
                                                'eA+A4QvwbiN0DMBMTcQLwUiAGDGwnx'
                                            ),
                                        },
                                    ],
                                ],
                                [
                                    'stable',
                                    [
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjZMQEAADmABU=',
                                        },
                                        {
                                            'array': '|b1',
                                            'shape': [20],
                                            'zlib64': 'eNpjZMQEAADmABU=',
                                        },
                                    ],
                                ],
                            ],
                        },
                    },
                },
            },
        },
        'agree': True,
    },
    'recurrent_rule.hmr_settled': {
        'platforms': {
            'arm64': {
                'research': '7440581b174be13887db6c391e45efc9c3ee2668',
                'recorded': '2026-10-22',
                'jax': '0.6.1',
                'jaxlib': '0.6.1',
                'numpy': '1.26.4',
                'xla_flags': ['', '--xla_cpu_multi_thread_eigen=false'],
                'values': {
                    '70d1f24b9604fd60e2bdbde501a926d569ed6526c66f1a4df00cf4d599500141': {
                        'tests': [
                            (
                                'test_recurrent_rule.py::TestHmrSettlingRule::'
                                'test_matches_hmrs_forward[None]'
                            ),
                            (
                                'test_recurrent_rule.py::TestHmrSettlingRule::'
                                'test_emrs_rule_is_a_different_network'
                            ),
                        ],
                        'inputs': [
                            [
                                '<f4[30, 3, 340]',
                                '<f4[30, 340, 1]',
                                '<i4[30, 10000]',
                                '<i4[30, 10000]',
                                '<f4[30, 10000]',
                                '|b1[30, 10000]',
                                '<f4[4, 3]',
                                '<f4[4, 1]',
                                18,
                            ],
                            {},
                        ],
                        'value': {
                            'array': '<f4',
                            'shape': [30],
                            'zlib64': (
                                'eNo7a8hgv4yf1Z6BgcH+koSk/SMGLjDbPE8STMfwMdpHyXLb7/jMYx/Qzm0fAxT7x8'
                                'NoX6ftYT9DgN++2ZfLvqaUyZ5nvaq9jSqXffI9dvtvSpb2Che47ZPucNvPYWWwF3Jn'
                                'sffbyWjPBNS7S57JXolZ2h4A3w8ddQ=='
                            ),
                        },
                    },
                    'a2fc11789d099210dc9c4e07cecb36b3ea76e1d8dd7a3c0f66073ec8feedc486': {
                        'tests': [
                            (
                                'test_recurrent_rule.py::TestHmrSettlingRule::'
                                'test_matches_hmrs_forward[dynamic_functions2]'
                            ),
                        ],
                        'inputs': [
                            [
                                '<f4[30, 3, 340]',
                                '<f4[30, 340, 1]',
                                '<i4[30, 10000]',
                                '<i4[30, 10000]',
                                '<f4[30, 10000]',
                                '|b1[30, 10000]',
                                '<f4[4, 3]',
                                '<f4[4, 1]',
                                18,
                            ],
                            {
                                'dynamic_functions': {
                                    'mode': 'cppn_output',
                                    'num_activations': 4,
                                    'hh_activation_mode': 'every_iteration',
                                },
                                'act_indices': '<i4[30, 340]',
                                'num_activations': 4,
                                'palette': None,
                            },
                        ],
                        'value': {
                            'array': '<f4',
                            'shape': [30],
                            'zlib64': (
                                'eNpjYGCwr+dlsGcA0rH8CmAahpmg9IIFRnAx0Sh7MP3FxxJIN9i9QlIPwlssWew7uv'
                                '7YgdjbPvCgyE1baQwxj5nBHgAzGRRE'
                            ),
                        },
                    },
                    'bc0ac242db169a346fc0bc7a19408b10cafe76daab84a5f472c8aefc9355423c': {
                        'tests': [
                            (
                                'test_recurrent_rule.py::TestHmrSettlingRule::'
                                'test_matches_hmrs_forward[dynamic_functions1]'
                            ),
                        ],
                        'inputs': [
                            [
                                '<f4[30, 3, 340]',
                                '<f4[30, 340, 1]',
                                '<i4[30, 10000]',
                                '<i4[30, 10000]',
                                '<f4[30, 10000]',
                                '|b1[30, 10000]',
                                '<f4[4, 3]',
                                '<f4[4, 1]',
                                18,
                            ],
                            {
                                'dynamic_functions': {
                                    'mode': 'cppn_output',
                                    'num_activations': 4,
                                    'hh_activation_mode': 'initial_only',
                                },
                                'act_indices': '<i4[30, 340]',
                                'num_activations': 4,
                                'palette': None,
                            },
                        ],
                        'value': {
                            'array': '<f4',
                            'shape': [30],
                            'zlib64': (
                                'eNpbLchoX8/LYP/dwNY+ll/BnoGBAYz1b3Db5zxSBrMXLDCyX1/JCmYLufLbTxIVtv'
                                '/iY2kvwNRg91eeESx+4a8AmG7ebmHvsEXZfk88o/22Dzxw8w5c4bFXu20MZh9jYLYH'
                                'AB3LHtY='
                            ),
                        },
                    },
                },
            },
            'x86_64': {
                'research': '7440581b1',
                'recorded': '2026-10-23',
                'jax': '0.6.1',
                'jaxlib': '0.6.1',
                'numpy': '1.26.4',
                'xla_flags': ['', '--xla_cpu_multi_thread_eigen=false'],
                'values': {
                    '14d3766e932d008a97678c8b8382980debb417a23874ed619b50ef656833cd1a': {
                        'tests': [
                            (
                                'test_recurrent_rule.py::TestHmrSettlingRule::'
                                'test_matches_hmrs_forward[dynamic_functions1]'
                            ),
                        ],
                        'inputs': [
                            [
                                '<f4[30, 3, 340]',
                                '<f4[30, 340, 1]',
                                '<i4[30, 10000]',
                                '<i4[30, 10000]',
                                '<f4[30, 10000]',
                                '|b1[30, 10000]',
                                '<f4[4, 3]',
                                '<f4[4, 1]',
                                18,
                            ],
                            {
                                'dynamic_functions': {
                                    'mode': 'cppn_output',
                                    'num_activations': 4,
                                    'hh_activation_mode': 'initial_only',
                                },
                                'act_indices': '<i4[30, 340]',
                                'num_activations': 4,
                                'palette': None,
                            },
                        ],
                        'value': {
                            'array': '<f4',
                            'shape': [30],
                            'zlib64': (
                                'eNpbLchoX8/LYP/dwNY+ll/BnoGBAYz1b3Db5zxSBrMXLDCyX1/JCmYLufLbTxIVtv'
                                '/iY2kvwNRg91eeESx+4a8AmG7ebmHvsEXZfk88o/22Dzxw8w5c4bFXu20MZh9jYLYH'
                                'AB3LHtY='
                            ),
                        },
                    },
                    '3f0da266e06ae182a5ba748c3fdac3c4b10f67891f05e2c2f4fbb84a93bd0662': {
                        'tests': [
                            (
                                'test_recurrent_rule.py::TestHmrSettlingRule::'
                                'test_matches_hmrs_forward[dynamic_functions2]'
                            ),
                        ],
                        'inputs': [
                            [
                                '<f4[30, 3, 340]',
                                '<f4[30, 340, 1]',
                                '<i4[30, 10000]',
                                '<i4[30, 10000]',
                                '<f4[30, 10000]',
                                '|b1[30, 10000]',
                                '<f4[4, 3]',
                                '<f4[4, 1]',
                                18,
                            ],
                            {
                                'dynamic_functions': {
                                    'mode': 'cppn_output',
                                    'num_activations': 4,
                                    'hh_activation_mode': 'every_iteration',
                                },
                                'act_indices': '<i4[30, 340]',
                                'num_activations': 4,
                                'palette': None,
                            },
                        ],
                        'value': {
                            'array': '<f4',
                            'shape': [30],
                            'zlib64': (
                                'eNpjYGCwr+dlsGcA0rH8CmAahpmg9IIFRnAx0Sh7MP3FxxJIN9i9QlIPwlssWew7uv'
                                '7YgdjbPvCgyE1baQwxj5nBHgAzGRRE'
                            ),
                        },
                    },
                    '70d1f24b9604fd60e2bdbde501a926d569ed6526c66f1a4df00cf4d599500141': {
                        'tests': [
                            (
                                'test_recurrent_rule.py::TestHmrSettlingRule::'
                                'test_matches_hmrs_forward[None]'
                            ),
                            (
                                'test_recurrent_rule.py::TestHmrSettlingRule::'
                                'test_emrs_rule_is_a_different_network'
                            ),
                        ],
                        'inputs': [
                            [
                                '<f4[30, 3, 340]',
                                '<f4[30, 340, 1]',
                                '<i4[30, 10000]',
                                '<i4[30, 10000]',
                                '<f4[30, 10000]',
                                '|b1[30, 10000]',
                                '<f4[4, 3]',
                                '<f4[4, 1]',
                                18,
                            ],
                            {},
                        ],
                        'value': {
                            'array': '<f4',
                            'shape': [30],
                            'zlib64': (
                                'eNo7a8hgv4yf1Z6BgcH+koSk/SMGLjDbPE8STMfwMdpHyXLb7/jMYx/Qzm0fAxT7x8'
                                'NoX6ftYT9DgN++2ZfLvqaUyZ5nvaq9jSqXffI9dvtvSpb2Che47ZPucNvPYWWwF3Jn'
                                'sffbyWjPBNS7S57JXolZ2h4A3w8ddQ=='
                            ),
                        },
                    },
                    'a2fc11789d099210dc9c4e07cecb36b3ea76e1d8dd7a3c0f66073ec8feedc486': {
                        'tests': [
                            (
                                'test_recurrent_rule.py::TestHmrSettlingRule::'
                                'test_matches_hmrs_forward[dynamic_functions2]'
                            ),
                        ],
                        'inputs': [
                            [
                                '<f4[30, 3, 340]',
                                '<f4[30, 340, 1]',
                                '<i4[30, 10000]',
                                '<i4[30, 10000]',
                                '<f4[30, 10000]',
                                '|b1[30, 10000]',
                                '<f4[4, 3]',
                                '<f4[4, 1]',
                                18,
                            ],
                            {
                                'dynamic_functions': {
                                    'mode': 'cppn_output',
                                    'num_activations': 4,
                                    'hh_activation_mode': 'every_iteration',
                                },
                                'act_indices': '<i4[30, 340]',
                                'num_activations': 4,
                                'palette': None,
                            },
                        ],
                        'value': {
                            'array': '<f4',
                            'shape': [30],
                            'zlib64': (
                                'eNpjYGCwr+dlsGcA0rH8CmAahpmg9IIFRnAx0Sh7MP3FxxJIN9i9QlIPwlssWew7uv'
                                '7YgdjbPvCgyE1baQwxj5nBHgAzGRRE'
                            ),
                        },
                    },
                    'bc0ac242db169a346fc0bc7a19408b10cafe76daab84a5f472c8aefc9355423c': {
                        'tests': [
                            (
                                'test_recurrent_rule.py::TestHmrSettlingRule::'
                                'test_matches_hmrs_forward[dynamic_functions1]'
                            ),
                        ],
                        'inputs': [
                            [
                                '<f4[30, 3, 340]',
                                '<f4[30, 340, 1]',
                                '<i4[30, 10000]',
                                '<i4[30, 10000]',
                                '<f4[30, 10000]',
                                '|b1[30, 10000]',
                                '<f4[4, 3]',
                                '<f4[4, 1]',
                                18,
                            ],
                            {
                                'dynamic_functions': {
                                    'mode': 'cppn_output',
                                    'num_activations': 4,
                                    'hh_activation_mode': 'initial_only',
                                },
                                'act_indices': '<i4[30, 340]',
                                'num_activations': 4,
                                'palette': None,
                            },
                        ],
                        'value': {
                            'array': '<f4',
                            'shape': [30],
                            'zlib64': (
                                'eNpbLchoX8/LYP/dwNY+ll/BnoGBAYz1b3Db5zxSBrMXLDCyX1/JCmYLufLbTxIVtv'
                                '/iY2kvwNRg91eeESx+4a8AmG7ebmHvsEXZfk88o/22Dzxw8w5c4bFXu20MZh9jYLYH'
                                'AB3LHtY='
                            ),
                        },
                    },
                    'e24a0a02d846ca443be3c74d671869947fd573077624a0393f5831c1f73cb95d': {
                        'tests': [
                            (
                                'test_recurrent_rule.py::TestHmrSettlingRule::'
                                'test_matches_hmrs_forward[None]'
                            ),
                            (
                                'test_recurrent_rule.py::TestHmrSettlingRule::'
                                'test_emrs_rule_is_a_different_network'
                            ),
                        ],
                        'inputs': [
                            [
                                '<f4[30, 3, 340]',
                                '<f4[30, 340, 1]',
                                '<i4[30, 10000]',
                                '<i4[30, 10000]',
                                '<f4[30, 10000]',
                                '|b1[30, 10000]',
                                '<f4[4, 3]',
                                '<f4[4, 1]',
                                18,
                            ],
                            {},
                        ],
                        'value': {
                            'array': '<f4',
                            'shape': [30],
                            'zlib64': (
                                'eNo7a8hgv4yf1Z6BgcH+koSk/SMGLjDbPE8STMfwMdpHyXLb7/jMYx/Qzm0fAxT7x8'
                                'NoX6ftYT9DgN++2ZfLvqaUyZ5nvaq9jSqXffI9dvtvSpb2Che47ZPucNvPYWWwF3Jn'
                                'sffbyWjPBNS7S57JXolZ2h4A3w8ddQ=='
                            ),
                        },
                    },
                },
            },
        },
        'agree': False,
    },
    'unified_recipe.run_uni': {
        'platforms': {
            'arm64': {
                'research': '7440581b174be13887db6c391e45efc9c3ee2668',
                'recorded': '2026-10-22',
                'jax': '0.6.1',
                'jaxlib': '0.6.1',
                'numpy': '1.26.4',
                'xla_flags': ['', '--xla_cpu_multi_thread_eigen=false'],
                'values': {
                    '2bcba8db35a236271e0c9ce7e66b08d5e9050c2b07eb2d847b0abf79b75e570b': {
                        'tests': [
                            (
                                'test_unified_recipe.py::TestRecipeAgainstLiveUni::'
                                'test_with_the_timed_cache'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'max_depth': 2,
                                            'variance_threshold': 0.0,
                                            'recurrence': {
                                                'allow_hidden_to_hidden': True,
                                                'hh_cache_enabled': True,
                                                'hh_refresh_interval': 2,
                                            },
                                        },
                                        'neat_species': {
                                            'compatibility_threshold': 2.5,
                                            'max_stagnation': 40,
                                        },
                                    },
                                },
                            },
                            {
                                'problem': '_TwoInputXOR',
                                'get_data': [
                                    [[0.0, 0.0], [0.0]],
                                    [[0.0, 1.0], [1.0]],
                                    [[1.0, 0.0], [1.0]],
                                    [[1.0, 1.0], [0.0]],
                                ],
                                'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                                'input_shape': [2],
                                'inputs': [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
                                'jitable': True,
                                'output_coords': [[0.0, 1.0]],
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                            },
                            42,
                            5,
                        ],
                        'value': {
                            'tuple': [
                                [
                                    {
                                        'array': '<f4',
                                        'shape': [20],
                                        'zlib64': 'eNpjYHhgx8BgYA/BIDaEXyKLTRwhj4khcgCPqxAU',
                                    },
                                    {
                                        'array': '<f4',
                                        'shape': [20],
                                        'zlib64': 'eNpjYDCwZ4DjB3YIjMyHsbGpQdUPAGfcD4M=',
                                    },
                                    {
                                        'array': '<f4',
                                        'shape': [20],
                                        'zlib64': 'eNpjYHhgxwDHBvYIGhtGqC2RRahb5/kQLg4Aij8Ruw==',
                                    },
                                    {
                                        'array': '<f4',
                                        'shape': [20],
                                        'zlib64': (
                                            'eNpjYDCwZ2B4YAfBBvb/zBFsEM4QZ7dnE3gN5cPkEPII/MBOBIgBzd'
                                            'IRQw=='
                                        ),
                                    },
                                    {
                                        'array': '<f4',
                                        'shape': [20],
                                        'zlib64': (
                                            'eNpjYDCwZ2B4YAfDJbIgGiSGitd5PkQTR+hBFgMAmzUQXQ=='
                                        ),
                                    },
                                ],
                                [
                                    {'tuple': [73584, 'refresh']},
                                    {'tuple': [73584, 'cached']},
                                    {'tuple': [73459, 'refresh']},
                                    {'tuple': [73459, 'cached']},
                                    {'tuple': [73414, 'refresh']},
                                ],
                            ],
                        },
                    },
                    '5051298d1542a50ac9aae91c5baa96ee28c07ef96d3d0ce64d263388ba9d4418': {
                        'tests': [
                            (
                                'test_unified_recipe.py::TestRecipeAgainstLiveUni::'
                                'test_without_the_cache'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'max_depth': 2,
                                            'variance_threshold': 0.0,
                                            'recurrence': {
                                                'allow_hidden_to_hidden': True,
                                                'hh_cache_enabled': False,
                                            },
                                        },
                                        'neat_species': {
                                            'compatibility_threshold': 2.5,
                                            'max_stagnation': 40,
                                        },
                                    },
                                },
                            },
                            {
                                'problem': '_TwoInputXOR',
                                'get_data': [
                                    [[0.0, 0.0], [0.0]],
                                    [[0.0, 1.0], [1.0]],
                                    [[1.0, 0.0], [1.0]],
                                    [[1.0, 1.0], [0.0]],
                                ],
                                'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                                'input_shape': [2],
                                'inputs': [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
                                'jitable': True,
                                'output_coords': [[0.0, 1.0]],
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                            },
                            42,
                            4,
                        ],
                        'value': {
                            'tuple': [
                                [
                                    {
                                        'array': '<f4',
                                        'shape': [20],
                                        'zlib64': 'eNpjYHhgx8BgYA/BIDaEXyKLTRwhj4khcgCPqxAU',
                                    },
                                    {
                                        'array': '<f4',
                                        'shape': [20],
                                        'zlib64': 'eNpjYDCwZyACl8g+sGNgQMcgOVQ+AOfyDrY=',
                                    },
                                    {
                                        'array': '<f4',
                                        'shape': [20],
                                        'zlib64': 'eNpjYDCwZyABl8g+sGNgQMYgcRhtYA8AsNEMqQ==',
                                    },
                                    {
                                        'array': '<f4',
                                        'shape': [20],
                                        'zlib64': 'eNpjYDCwZ6AIP7AD0QVQPgBjZgnM',
                                    },
                                ],
                                [
                                    {'tuple': [73584, 'refresh']},
                                    {'tuple': [73566, 'refresh']},
                                    {'tuple': [73540, 'refresh']},
                                    {'tuple': [73807, 'refresh']},
                                ],
                            ],
                        },
                    },
                    '5c5fa9c8cce8b112d39996823fecc72f25dffb1c32d301fa5a4c2ecbf2e1b9dc': {
                        'tests': [
                            (
                                'test_unified_recipe.py::TestRecipeAgainstLiveUni::'
                                'test_each_part_of_the_recipe_matters[None-None-40]'
                            ),
                            (
                                'test_unified_recipe.py::TestRecipeAgainstLiveUni::'
                                'test_each_part_of_the_recipe_matters[emr_overrides4-recurrence_ove'
                                'rrides4-40]'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 40,
                                        'substrate': {
                                            'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'max_depth': 2,
                                            'variance_threshold': 0.0,
                                            'recurrence': {
                                                'allow_hidden_to_hidden': True,
                                                'hh_cache_enabled': False,
                                            },
                                            'band_threshold': 1.0,
                                        },
                                        'neat_species': {
                                            'compatibility_threshold': 2.5,
                                            'max_stagnation': 40,
                                        },
                                    },
                                },
                            },
                            {
                                'problem': '_TwoInputXOR',
                                'get_data': [
                                    [[0.0, 0.0], [0.0]],
                                    [[0.0, 1.0], [1.0]],
                                    [[1.0, 0.0], [1.0]],
                                    [[1.0, 1.0], [0.0]],
                                ],
                                'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                                'input_shape': [2],
                                'inputs': [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
                                'jitable': True,
                                'output_coords': [[0.0, 1.0]],
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                            },
                            42,
                            2,
                        ],
                        'value': {
                            'tuple': [
                                [
                                    {
                                        'array': '<f4',
                                        'shape': [40],
                                        'zlib64': (
                                            'eNpjYHhgx8BgYA/BIDaEn8KLTRwhj64eUw0uPch6DexX5GM3bwc7hA'
                                            'YA5nQjWQ=='
                                        ),
                                    },
                                    {
                                        'array': '<f4',
                                        'shape': [40],
                                        'zlib64': 'eNpjYDCwZ6AYP7AjLFeARR2IDxODsZGxgT0AVdwYbw==',
                                    },
                                ],
                                [{'tuple': [141798, 'refresh']}, {'tuple': [144440, 'refresh']}],
                            ],
                        },
                    },
                    'b2629a366ea07901d46b9278a1c8bec745cc18b35b4d4e788d996e3957ea7b60': {
                        'tests': [
                            (
                                'test_unified_recipe.py::TestRecipeAgainstLiveUni::'
                                'test_each_part_of_the_recipe_matters[None-None-20]'
                            ),
                            (
                                'test_unified_recipe.py::TestRecipeAgainstLiveUni::'
                                'test_each_part_of_the_recipe_matters[emr_overrides2-recurrence_ove'
                                'rrides2-20]'
                            ),
                            (
                                'test_unified_recipe.py::TestRecipeAgainstLiveUni::'
                                'test_each_part_of_the_recipe_matters[emr_overrides3-recurrence_ove'
                                'rrides3-20]'
                            ),
                            (
                                'test_unified_recipe.py::TestRecipeAgainstLiveUni::'
                                'test_each_part_of_the_recipe_matters[emr_overrides5-recurrence_ove'
                                'rrides5-20]'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'max_depth': 2,
                                            'variance_threshold': 0.0,
                                            'recurrence': {
                                                'allow_hidden_to_hidden': True,
                                                'hh_cache_enabled': False,
                                            },
                                            'band_threshold': 1.0,
                                        },
                                        'neat_species': {
                                            'compatibility_threshold': 2.5,
                                            'max_stagnation': 40,
                                        },
                                    },
                                },
                            },
                            {
                                'problem': '_TwoInputXOR',
                                'get_data': [
                                    [[0.0, 0.0], [0.0]],
                                    [[0.0, 1.0], [1.0]],
                                    [[1.0, 0.0], [1.0]],
                                    [[1.0, 1.0], [0.0]],
                                ],
                                'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                                'input_shape': [2],
                                'inputs': [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
                                'jitable': True,
                                'output_coords': [[0.0, 1.0]],
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                            },
                            42,
                            2,
                        ],
                        'value': {
                            'tuple': [
                                [
                                    {
                                        'array': '<f4',
                                        'shape': [20],
                                        'zlib64': 'eNpjYHhgx8BgYA/BIDaEn8KLTRwhj64exgYAnp0RUg==',
                                    },
                                    {
                                        'array': '<f4',
                                        'shape': [20],
                                        'zlib64': 'eNpjYDCwZ8CDU3gf2DEw4MIgNTAaggEX0Q6W',
                                    },
                                ],
                                [{'tuple': [70965, 'refresh']}, {'tuple': [71308, 'refresh']}],
                            ],
                        },
                    },
                },
            },
            'x86_64': {
                'research': '7440581b1',
                'recorded': '2026-10-23',
                'jax': '0.6.1',
                'jaxlib': '0.6.1',
                'numpy': '1.26.4',
                'xla_flags': ['', '--xla_cpu_multi_thread_eigen=false'],
                'values': {
                    '2bcba8db35a236271e0c9ce7e66b08d5e9050c2b07eb2d847b0abf79b75e570b': {
                        'tests': [
                            (
                                'test_unified_recipe.py::TestRecipeAgainstLiveUni::'
                                'test_with_the_timed_cache'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'max_depth': 2,
                                            'variance_threshold': 0.0,
                                            'recurrence': {
                                                'allow_hidden_to_hidden': True,
                                                'hh_cache_enabled': True,
                                                'hh_refresh_interval': 2,
                                            },
                                        },
                                        'neat_species': {
                                            'compatibility_threshold': 2.5,
                                            'max_stagnation': 40,
                                        },
                                    },
                                },
                            },
                            {
                                'problem': '_TwoInputXOR',
                                'get_data': [
                                    [[0.0, 0.0], [0.0]],
                                    [[0.0, 1.0], [1.0]],
                                    [[1.0, 0.0], [1.0]],
                                    [[1.0, 1.0], [0.0]],
                                ],
                                'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                                'input_shape': [2],
                                'inputs': [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
                                'jitable': True,
                                'output_coords': [[0.0, 1.0]],
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                            },
                            42,
                            5,
                        ],
                        'value': {
                            'tuple': [
                                [
                                    {
                                        'array': '<f4',
                                        'shape': [20],
                                        'zlib64': 'eNpjYHhgx8BgYA/BIDaEXyKLTRwhj4khcgCPqxAU',
                                    },
                                    {
                                        'array': '<f4',
                                        'shape': [20],
                                        'zlib64': 'eNpjYDCwZ4DjB3YIjMyHsbGpQdUPAGfcD4M=',
                                    },
                                    {
                                        'array': '<f4',
                                        'shape': [20],
                                        'zlib64': 'eNpjYHhgxwDHBvYIGhtGqC2RRahb5/kQLg4Aij8Ruw==',
                                    },
                                    {
                                        'array': '<f4',
                                        'shape': [20],
                                        'zlib64': (
                                            'eNpjYDCwZ2B4YAfBBvb/zBFsEM4TZ7dnE3gN5cPkEPII/MBOBIgBzv'
                                            'IRSQ=='
                                        ),
                                    },
                                    {
                                        'array': '<f4',
                                        'shape': [20],
                                        'zlib64': (
                                            'eNpjYDCwZ2B4YAfDJbIgGiSGitd5PkQTR+hBFgMAmzUQXQ=='
                                        ),
                                    },
                                ],
                                [
                                    {'tuple': [73584, 'refresh']},
                                    {'tuple': [73584, 'cached']},
                                    {'tuple': [73459, 'refresh']},
                                    {'tuple': [73459, 'cached']},
                                    {'tuple': [73414, 'refresh']},
                                ],
                            ],
                        },
                    },
                    '5051298d1542a50ac9aae91c5baa96ee28c07ef96d3d0ce64d263388ba9d4418': {
                        'tests': [
                            (
                                'test_unified_recipe.py::TestRecipeAgainstLiveUni::'
                                'test_without_the_cache'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'max_depth': 2,
                                            'variance_threshold': 0.0,
                                            'recurrence': {
                                                'allow_hidden_to_hidden': True,
                                                'hh_cache_enabled': False,
                                            },
                                        },
                                        'neat_species': {
                                            'compatibility_threshold': 2.5,
                                            'max_stagnation': 40,
                                        },
                                    },
                                },
                            },
                            {
                                'problem': '_TwoInputXOR',
                                'get_data': [
                                    [[0.0, 0.0], [0.0]],
                                    [[0.0, 1.0], [1.0]],
                                    [[1.0, 0.0], [1.0]],
                                    [[1.0, 1.0], [0.0]],
                                ],
                                'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                                'input_shape': [2],
                                'inputs': [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
                                'jitable': True,
                                'output_coords': [[0.0, 1.0]],
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                            },
                            42,
                            4,
                        ],
                        'value': {
                            'tuple': [
                                [
                                    {
                                        'array': '<f4',
                                        'shape': [20],
                                        'zlib64': 'eNpjYHhgx8BgYA/BIDaEXyKLTRwhj4khcgCPqxAU',
                                    },
                                    {
                                        'array': '<f4',
                                        'shape': [20],
                                        'zlib64': 'eNpjYDCwZyACl8g+sGNgQMcgOVQ+AOfyDrY=',
                                    },
                                    {
                                        'array': '<f4',
                                        'shape': [20],
                                        'zlib64': 'eNpjYDCwZyABl8g+sGNgQMYgcRhtYA8AsNEMqQ==',
                                    },
                                    {
                                        'array': '<f4',
                                        'shape': [20],
                                        'zlib64': 'eNpjYDCwZ6AIP7AD0QVQPgBjZgnM',
                                    },
                                ],
                                [
                                    {'tuple': [73584, 'refresh']},
                                    {'tuple': [73566, 'refresh']},
                                    {'tuple': [73540, 'refresh']},
                                    {'tuple': [73807, 'refresh']},
                                ],
                            ],
                        },
                    },
                    '5c5fa9c8cce8b112d39996823fecc72f25dffb1c32d301fa5a4c2ecbf2e1b9dc': {
                        'tests': [
                            (
                                'test_unified_recipe.py::TestRecipeAgainstLiveUni::'
                                'test_each_part_of_the_recipe_matters[None-None-40]'
                            ),
                            (
                                'test_unified_recipe.py::TestRecipeAgainstLiveUni::'
                                'test_each_part_of_the_recipe_matters[emr_overrides4-recurrence_ove'
                                'rrides4-40]'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 40,
                                        'substrate': {
                                            'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'max_depth': 2,
                                            'variance_threshold': 0.0,
                                            'recurrence': {
                                                'allow_hidden_to_hidden': True,
                                                'hh_cache_enabled': False,
                                            },
                                            'band_threshold': 1.0,
                                        },
                                        'neat_species': {
                                            'compatibility_threshold': 2.5,
                                            'max_stagnation': 40,
                                        },
                                    },
                                },
                            },
                            {
                                'problem': '_TwoInputXOR',
                                'get_data': [
                                    [[0.0, 0.0], [0.0]],
                                    [[0.0, 1.0], [1.0]],
                                    [[1.0, 0.0], [1.0]],
                                    [[1.0, 1.0], [0.0]],
                                ],
                                'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                                'input_shape': [2],
                                'inputs': [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
                                'jitable': True,
                                'output_coords': [[0.0, 1.0]],
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                            },
                            42,
                            2,
                        ],
                        'value': {
                            'tuple': [
                                [
                                    {
                                        'array': '<f4',
                                        'shape': [40],
                                        'zlib64': (
                                            'eNpjYHhgx8BgYA/BIDaEn8KLTRwhj64eUw0uPch6DexX5GM3bwc7hA'
                                            'YA5nQjWQ=='
                                        ),
                                    },
                                    {
                                        'array': '<f4',
                                        'shape': [40],
                                        'zlib64': 'eNpjYDCwZ6AYP7AjLFeARR2IDxODsZGxgT0AVdwYbw==',
                                    },
                                ],
                                [{'tuple': [141798, 'refresh']}, {'tuple': [144440, 'refresh']}],
                            ],
                        },
                    },
                    'b2629a366ea07901d46b9278a1c8bec745cc18b35b4d4e788d996e3957ea7b60': {
                        'tests': [
                            (
                                'test_unified_recipe.py::TestRecipeAgainstLiveUni::'
                                'test_each_part_of_the_recipe_matters[None-None-20]'
                            ),
                            (
                                'test_unified_recipe.py::TestRecipeAgainstLiveUni::'
                                'test_each_part_of_the_recipe_matters[emr_overrides2-recurrence_ove'
                                'rrides2-20]'
                            ),
                            (
                                'test_unified_recipe.py::TestRecipeAgainstLiveUni::'
                                'test_each_part_of_the_recipe_matters[emr_overrides3-recurrence_ove'
                                'rrides3-20]'
                            ),
                            (
                                'test_unified_recipe.py::TestRecipeAgainstLiveUni::'
                                'test_each_part_of_the_recipe_matters[emr_overrides5-recurrence_ove'
                                'rrides5-20]'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'max_depth': 2,
                                            'variance_threshold': 0.0,
                                            'recurrence': {
                                                'allow_hidden_to_hidden': True,
                                                'hh_cache_enabled': False,
                                            },
                                            'band_threshold': 1.0,
                                        },
                                        'neat_species': {
                                            'compatibility_threshold': 2.5,
                                            'max_stagnation': 40,
                                        },
                                    },
                                },
                            },
                            {
                                'problem': '_TwoInputXOR',
                                'get_data': [
                                    [[0.0, 0.0], [0.0]],
                                    [[0.0, 1.0], [1.0]],
                                    [[1.0, 0.0], [1.0]],
                                    [[1.0, 1.0], [0.0]],
                                ],
                                'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                                'input_shape': [2],
                                'inputs': [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
                                'jitable': True,
                                'output_coords': [[0.0, 1.0]],
                                'output_shape': [1],
                                'targets': [[0.0], [1.0], [1.0], [0.0]],
                            },
                            42,
                            2,
                        ],
                        'value': {
                            'tuple': [
                                [
                                    {
                                        'array': '<f4',
                                        'shape': [20],
                                        'zlib64': 'eNpjYHhgx8BgYA/BIDaEn8KLTRwhj64exgYAnp0RUg==',
                                    },
                                    {
                                        'array': '<f4',
                                        'shape': [20],
                                        'zlib64': 'eNpjYDCwZ8CDU3gf2DEw4MIgNTAaggEX0Q6W',
                                    },
                                ],
                                [{'tuple': [70965, 'refresh']}, {'tuple': [71308, 'refresh']}],
                            ],
                        },
                    },
                },
            },
        },
        'agree': False,
    },
    'unified_recipe.uni_flags': {
        'platforms': {
            'arm64': {
                'research': '7440581b174be13887db6c391e45efc9c3ee2668',
                'recorded': '2026-10-22',
                'jax': '0.6.1',
                'jaxlib': '0.6.1',
                'numpy': '1.26.4',
                'xla_flags': ['', '--xla_cpu_multi_thread_eigen=false'],
                'values': {
                    '104573ccca5baef29ede2bb0eebb8df4cf65863230d54312d3d1c02f230357d7': {
                        'tests': [
                            (
                                'test_unified_recipe.py::TestUniNames::'
                                'test_uni_sets_the_same_rule[recurrence0-flags0]'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'max_depth': 2,
                                            'variance_threshold': 0.0,
                                            'recurrence': {
                                                'enabled': True,
                                                'allow_hidden_to_hidden': True,
                                                'allow_lateral': True,
                                            },
                                        },
                                        'neat_species': {
                                            'compatibility_threshold': 2.5,
                                            'max_stagnation': 40,
                                        },
                                    },
                                },
                            },
                        ],
                        'value': {'tuple': [True, True]},
                    },
                    '59cb3a8b632c0f4295127015b61bacf919fbf1d46760eeae7ca00d9ebffeb938': {
                        'tests': [
                            (
                                'test_unified_recipe.py::TestUniNames::'
                                'test_uni_sets_the_same_rule[recurrence1-flags1]'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'max_depth': 2,
                                            'variance_threshold': 0.0,
                                            'recurrence': {'allow_hidden_to_hidden': True},
                                        },
                                        'neat_species': {
                                            'compatibility_threshold': 2.5,
                                            'max_stagnation': 40,
                                        },
                                    },
                                },
                            },
                        ],
                        'value': {'tuple': [True, False]},
                    },
                },
            },
            'x86_64': {
                'research': '7440581b1',
                'recorded': '2026-10-23',
                'jax': '0.6.1',
                'jaxlib': '0.6.1',
                'numpy': '1.26.4',
                'xla_flags': ['', '--xla_cpu_multi_thread_eigen=false'],
                'values': {
                    '104573ccca5baef29ede2bb0eebb8df4cf65863230d54312d3d1c02f230357d7': {
                        'tests': [
                            (
                                'test_unified_recipe.py::TestUniNames::'
                                'test_uni_sets_the_same_rule[recurrence0-flags0]'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'max_depth': 2,
                                            'variance_threshold': 0.0,
                                            'recurrence': {
                                                'enabled': True,
                                                'allow_hidden_to_hidden': True,
                                                'allow_lateral': True,
                                            },
                                        },
                                        'neat_species': {
                                            'compatibility_threshold': 2.5,
                                            'max_stagnation': 40,
                                        },
                                    },
                                },
                            },
                        ],
                        'value': {'tuple': [True, True]},
                    },
                    '59cb3a8b632c0f4295127015b61bacf919fbf1d46760eeae7ca00d9ebffeb938': {
                        'tests': [
                            (
                                'test_unified_recipe.py::TestUniNames::'
                                'test_uni_sets_the_same_rule[recurrence1-flags1]'
                            ),
                        ],
                        'inputs': [
                            {
                                'algorithm_params': {
                                    'hmrhyperneat': {
                                        'population_size': 20,
                                        'substrate': {
                                            'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                                            'output_coords': [[0.0, 1.0]],
                                        },
                                        'hmr_hyperneat': {
                                            'initial_depth': 0,
                                            'max_depth': 2,
                                            'variance_threshold': 0.0,
                                            'recurrence': {'allow_hidden_to_hidden': True},
                                        },
                                        'neat_species': {
                                            'compatibility_threshold': 2.5,
                                            'max_stagnation': 40,
                                        },
                                    },
                                },
                            },
                        ],
                        'value': {'tuple': [True, False]},
                    },
                },
            },
        },
        'agree': True,
    },
    'unified_recipe.uni_record_fields': {
        'platforms': {
            'arm64': {
                'research': '7440581b174be13887db6c391e45efc9c3ee2668',
                'recorded': '2026-10-22',
                'jax': '0.6.1',
                'jaxlib': '0.6.1',
                'numpy': '1.26.4',
                'xla_flags': ['', '--xla_cpu_multi_thread_eigen=false'],
                'values': {
                    '687f5fe8c114639c11f0accc6abd3993b11a0d36cb8d9b7d42e18707a444b798': {
                        'tests': [
                            (
                                'test_unified_recipe.py::TestUniNames::'
                                'test_the_record_carries_unis_fields'
                            ),
                        ],
                        'inputs': [],
                        'value': {
                            'set': [
                                'activate_time_used',
                                'avg_in_degree',
                                'avg_out_degree',
                                'backward_connections',
                                'fitness',
                                'forward_connections',
                                'forward_pass_ms',
                                'generations_to_solve',
                                'has_cycles',
                                'hidden_to_hidden_connections',
                                'lateral_connections',
                                'phase2_discovery_ms',
                                'self_loop_connections',
                                'substrate_discovery_ms',
                                'total_connections',
                            ],
                        },
                    },
                },
            },
            'x86_64': {
                'research': '7440581b1',
                'recorded': '2026-10-23',
                'jax': '0.6.1',
                'jaxlib': '0.6.1',
                'numpy': '1.26.4',
                'xla_flags': ['', '--xla_cpu_multi_thread_eigen=false'],
                'values': {
                    '687f5fe8c114639c11f0accc6abd3993b11a0d36cb8d9b7d42e18707a444b798': {
                        'tests': [
                            (
                                'test_unified_recipe.py::TestUniNames::'
                                'test_the_record_carries_unis_fields'
                            ),
                        ],
                        'inputs': [],
                        'value': {
                            'set': [
                                'activate_time_used',
                                'avg_in_degree',
                                'avg_out_degree',
                                'backward_connections',
                                'fitness',
                                'forward_connections',
                                'forward_pass_ms',
                                'generations_to_solve',
                                'has_cycles',
                                'hidden_to_hidden_connections',
                                'lateral_connections',
                                'phase2_discovery_ms',
                                'self_loop_connections',
                                'substrate_discovery_ms',
                                'total_connections',
                            ],
                        },
                    },
                },
            },
        },
        'agree': True,
    },
}
