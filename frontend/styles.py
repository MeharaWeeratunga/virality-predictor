"""
CSS styles for the application
"""


def get_custom_css():
    """Return custom CSS styles"""
    return """
    <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            color: #1f77b4;
            margin-bottom: 1rem;
        }
        .sub-header {
            text-align: center;
            color: #666;
            margin-bottom: 2rem;
        }
        .prediction-box {
            padding: 2rem;
            border-radius: 10px;
            border: 2px solid #e0e0e0;
            margin: 1rem 0;
        }
        .viral {
            background-color: #fff3cd;
            border-color: #ffc107;
        }
        .not-viral {
            background-color: #d1ecf1;
            border-color: #17a2b8;
        }
    </style>
    """