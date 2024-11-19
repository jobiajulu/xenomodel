import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from model import (Costs, RegionalDistribution, 
                  EnhancedXenoTransplantScaling, run_enhanced_analysis)

st.set_page_config(
    page_title="Xenotransplantation Impact Model",
    page_icon="🫀",
    layout="wide"
)

def main():
    # Title and introduction
    st.title("🫀 Xenotransplantation Impact Projection")
    st.markdown("""
    This model projects the potential impact of xenotransplantation on lives saved,
    based on various growth and implementation scenarios.
    """)

    # Sidebar for parameters
    with st.sidebar:
        st.header("Model Parameters")
        
        # Core scenario parameters
        scenario = st.selectbox(
            "Growth Scenario",
            options=["conservative", "moderate", "aggressive"],
            help="Affects the rate of xenotransplantation adoption and scaling"
        )
        
        years = st.slider(
            "Projection Years",
            min_value=10,
            max_value=50,
            value=30,
            help="Number of years to project into the future"
        )
        
        organ_type = st.selectbox(
            "Organ Type",
            options=["kidney", "heart", "liver", "lung", "pancreas"],
            help="Type of organ for transplantation"
        )

        # Advanced parameters collapsible
        with st.expander("Advanced Parameters"):
            initial_facilities = st.number_input(
                "Initial Facilities",
                min_value=1,
                max_value=10,
                value=2,
                help="Number of production facilities at start"
            )
            
            success_rate = st.slider(
                "Procedure Success Rate (%)",
                min_value=70,
                max_value=100,
                value=85,
                help="Expected success rate of xenotransplantation procedures"
            )
            
            learning_rate = st.slider(
                "Learning Rate",
                min_value=0.80,
                max_value=1.00,
                value=0.90,
                step=0.01,
                help="Cost reduction per doubling of production (0.9 = 10% reduction)"
            )

        # Cost parameters collapsible
        with st.expander("Cost Parameters"):
            facility_cost = st.number_input(
                "Facility Construction (Millions USD)",
                min_value=50,
                max_value=200,
                value=100,
                help="Cost to build a new production facility"
            )
            
            organ_processing = st.number_input(
                "Organ Processing Cost (USD)",
                min_value=10000,
                max_value=100000,
                value=50000,
                step=5000,
                help="Cost to process each organ"
            )

    # Run model with selected parameters
    costs = Costs(
        facility_construction=facility_cost * 1_000_000,
        facility_operation=20_000_000,
        organ_processing=organ_processing,
        transplant_procedure=250_000,
        training_per_team=500_000,
        cold_chain_per_mile=100
    )
    
    regions = RegionalDistribution()
    scaling_model = EnhancedXenoTransplantScaling(costs, regions)
    
    # Run analysis
    projections, metrics = run_enhanced_analysis(
        organ_type=organ_type,
        years=years,
        scenario=scenario
    )

    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        # Lives saved plot
        fig1 = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig1.add_trace(
            go.Bar(
                x=projections['year'],
                y=projections['annual_lives_saved'],
                name="Annual Lives Saved"
            ),
            secondary_y=False
        )
        
        fig1.add_trace(
            go.Scatter(
                x=projections['year'],
                y=projections['cumulative_lives_saved'],
                name="Cumulative Lives Saved",
                line=dict(width=3)
            ),
            secondary_y=True
        )
        
        fig1.update_layout(
            title="Lives Saved from Xenotransplantation",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Cost trajectory
        fig2 = go.Figure()
        
        fig2.add_trace(
            go.Scatter(
                x=projections['year'],
                y=projections['unit_cost'],
                name="Cost per Transplant",
                fill='tonexty'
            )
        )
        
        fig2.update_layout(
            title="Cost per Xenotransplant Over Time",
            yaxis_title="Cost (USD)",
            hovermode="x"
        )
        
        st.plotly_chart(fig2, use_container_width=True)

    # Summary metrics
    st.header("Impact Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Lives Saved",
            f"{metrics['cumulative_lives_saved']:,.0f}",
            delta=f"{projections['annual_lives_saved'].iloc[-1]:,.0f}/year"
        )
    
    with col2:
        st.metric(
            "Cost per QALY",
            f"${metrics['cost_per_qaly']:,.0f}"
        )
    
    with col3:
        st.metric(
            "Total Investment Required",
            f"${metrics['total_investment_required']/1e9:.1f}B"
        )
    
    with col4:
        st.metric(
            "Final Unit Cost",
            f"${metrics['final_unit_cost']:,.0f}",
            delta=f"{((metrics['final_unit_cost']/organ_processing)-1)*100:.1f}%"
        )

    # Regional distribution
    st.header("Regional Distribution")
    final_distribution = projections.iloc[-1]['regional_distribution']
    
    fig3 = go.Figure(data=[
        go.Bar(
            x=list(final_distribution.keys()),
            y=list(final_distribution.values())
        )
    ])
    
    fig3.update_layout(
        title="Final Year Regional Distribution",
        xaxis_title="Region",
        yaxis_title="Number of Transplants"
    )
    
    st.plotly_chart(fig3, use_container_width=True)

    # Export options
    st.header("Export Results")
    if st.button("Download Projection Data"):
        csv = projections.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="xenotransplant_projections.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()