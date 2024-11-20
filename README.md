# Xenotransplantation Impact Projection Model 🫀

## Overview
This project provides an interactive model for projecting the potential impact of xenotransplantation (animal-to-human organ transplantation) on saving lives over the next several decades. The model considers facility scaling, cost dynamics, regional distribution, and health outcomes to provide comprehensive projections.

## Live Demo
[Link to your Streamlit app once deployed]

![Screenshot of dashboard - you can add this later]

## Features
- **Interactive Projections**: Adjust key parameters and see results in real-time
- **Multiple Scenarios**: Compare conservative, moderate, and aggressive growth scenarios
- **Regional Analysis**: View impact distribution across different geographical regions
- **Cost Modeling**: Track cost trajectories with learning curve effects
- **Health Impact**: Calculate lives saved and quality-adjusted life years (QALYs)
- **Data Export**: Download projection data for further analysis

## Model Components

### Core Parameters
- Initial facility count and capacity
- Growth rate scenarios
- Procedure success rates
- Learning rate for cost reduction
- Regional distribution factors

### Key Metrics
- Annual and cumulative lives saved
- Cost per procedure over time
- Regional distribution of impact
- Required investment
- Cost per QALY
- Facility scaling trajectory

## Installation

### Prerequisites
- Python 3.8 or higher
- Git

### Local Setup
```bash
# Clone the repository
git clone https://github.com/jobiajulu/xenomodel.git
cd xenomodel

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app locally
streamlit run streamlit_app.py
```

## Usage

### Running the Model
1. Visit the deployed Streamlit app
2. Adjust parameters in the sidebar:
   - Select growth scenario
   - Set projection timeframe
   - Modify facility parameters
   - Adjust cost factors
3. View results in real-time across multiple visualizations
4. Download data for further analysis

### Parameter Definitions

#### Growth Scenarios
- **Conservative**: 20% year-over-year growth
- **Moderate**: 40% year-over-year growth
- **Aggressive**: 60% year-over-year growth

#### Cost Parameters
- **Facility Construction**: Capital cost for new production facility
- **Facility Operation**: Annual operating costs
- **Organ Processing**: Per-organ processing cost
- **Training**: Surgical team training costs

#### Health Impact
- **Success Rate**: Procedure success probability
- **Quality of Life**: Post-transplant quality of life adjustments
- **Survival Rates**: Mortality rate adjustments

## Model Methodology

### Facility Scaling
The model uses a compound growth approach with initial slow growth during regulatory approval and early adoption phases, followed by accelerated scaling based on the selected scenario.

```python
# Example scaling calculation
annual_supply = facilities * capacity * utilization_rate
facilities *= growth_rate[scenario]
```

### Cost Learning Curve
Implements Wright's Law for cost reduction through learning effects:
```python
unit_cost = base_cost * (cumulative_production ** log2(learning_rate))
```

### Health Impact Calculation
Combines survival probability with quality of life adjustments:
```python
qalys = survival_probability * qol_multiplier * years
```

## Project Structure
```
xenomodel/
├── README.md
├── requirements.txt
├── model.py              # Core model implementation
├── streamlit_app.py      # Interactive web interface
└── data/                 # (Optional) Reference data
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

### Development Process
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License
MIT License

Copyright (c) 2024 Joseph Obiajulu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Authors
Joseph Obiajulu  
NYU Grossman School of Medicine

## Citation
Please cite this work as:
```
Obiajulu J. Xenotransplantation Impact Projection Model [Computer software]. 
Version 1.0. New York, NY: NYU Grossman School of Medicine; 2024. 
Available at: [repository-url]
```

## Contact
Joseph Obiajulu  
NYU Grossman School of Medicine  
Email: joseph.obiajulu@nyulangone.org

---
> Note: This model is for research and projection purposes only. All medical decisions should be made in consultation with healthcare professionals.