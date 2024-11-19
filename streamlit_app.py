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
    projections, metrics, all_transplants, waitlist_deaths = run_enhanced_analysis(
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
                y=projections['total_supply'],
                name="Annual Lives Saved"
            ),
            secondary_y=False
        )
        
        fig1.add_trace(
            go.Scatter(
                x=projections['year'],
                y=projections['total_supply'].cumsum(),
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
            f"{projections['total_supply'].sum():,.0f}",
            delta=f"{projections['total_supply'].iloc[-1]:,.0f}/year"
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

    # Transplant Type Projections
    st.header("Transplant Type Projections")
    
    # Create the composite line graph
    fig4 = go.Figure()
    
    # Add traces for each transplant type
    for column in ['Xenotransplants', 'Deceased Donor', 'Living Donor']:
        fig4.add_trace(
            go.Scatter(
                x=all_transplants['year'],
                y=all_transplants[column],
                name=column,
                mode='lines',
                line=dict(width=3)
            )
        )
    
    # Add total line
    fig4.add_trace(
        go.Scatter(
            x=all_transplants['year'],
            y=all_transplants['Total'],
            name='Total Transplants',
            mode='lines',
            line=dict(width=3, dash='dash')
        )
    )
    
    fig4.update_layout(
        title=f"Projected {organ_type.title()} Transplants by Source",
        xaxis_title="Year",
        yaxis_title="Number of Transplants",
        hovermode="x unified",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    st.plotly_chart(fig4, use_container_width=True)
    
    # Add a summary table below the graph
    st.subheader("Transplant Projections Summary")
    summary_years = [0, 4, 9, 14, 19] if years > 20 else [0, 2, 4, 9]
    summary_df = all_transplants[all_transplants['year'].isin(summary_years)].copy()
    summary_df['year'] = summary_df['year'].apply(lambda x: f"Year {x}")
    st.dataframe(
        summary_df.set_index('year').round(0),
        use_container_width=True
    )

    # Waitlist Mortality Impact
    st.header("Waitlist Mortality Impact")
    
    # Create the mortality comparison graph
    fig5 = go.Figure()
    
    # Add traces for deaths with and without xenotransplantation
    fig5.add_trace(
        go.Scatter(
            x=waitlist_deaths['year'],
            y=waitlist_deaths['baseline_deaths'],
            name='Deaths without Xenotransplantation',
            mode='lines',
            line=dict(width=3, color='rgb(239, 85, 59)')
        )
    )
    
    fig5.add_trace(
        go.Scatter(
            x=waitlist_deaths['year'],
            y=waitlist_deaths['deaths_with_xeno'],
            name='Deaths with Xenotransplantation',
            mode='lines',
            line=dict(width=3, color='rgb(99, 110, 250)')
        )
    )
    
    # Add shaded area for lives saved
    fig5.add_trace(
        go.Scatter(
            x=waitlist_deaths['year'],
            y=waitlist_deaths['baseline_deaths'],
            fill=None,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        )
    )
    
    fig5.add_trace(
        go.Scatter(
            x=waitlist_deaths['year'],
            y=waitlist_deaths['deaths_with_xeno'],
            fill='tonexty',
            mode='lines',
            line=dict(width=0),
            name='Lives Saved',
            fillcolor='rgba(99, 110, 250, 0.2)'
        )
    )
    
    fig5.update_layout(
        title=f"Projected Annual Deaths on {organ_type.title()} Waitlist",
        xaxis_title="Year",
        yaxis_title="Number of Deaths",
        hovermode="x unified",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    st.plotly_chart(fig5, use_container_width=True)
    
    # Add summary metrics
    total_lives_saved = waitlist_deaths['lives_saved'].sum()
    final_year_reduction = (
        (waitlist_deaths['baseline_deaths'].iloc[-1] - 
         waitlist_deaths['deaths_with_xeno'].iloc[-1]) / 
        waitlist_deaths['baseline_deaths'].iloc[-1] * 100
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Total Lives Saved from Waitlist",
            f"{total_lives_saved:,.0f}",
            help="Cumulative reduction in waitlist deaths over the projection period"
        )
    with col2:
        st.metric(
            "Final Year Mortality Reduction",
            f"{final_year_reduction:.1f}%",
            help="Percentage reduction in annual deaths by the final year"
        )

    st.header("The Human Cost of Waiting")
    col1, col2 = st.columns(2)

    with col1:
        # Cumulative deaths over time
        fig_deaths = go.Figure()
        
        cumulative_baseline_deaths = waitlist_deaths['baseline_deaths'].cumsum()
        cumulative_xeno_deaths = waitlist_deaths['deaths_with_xeno'].cumsum()
        
        fig_deaths.add_trace(
            go.Scatter(
                x=waitlist_deaths['year'],
                y=cumulative_baseline_deaths,
                name='Deaths without Xenotransplantation',
                fill='tozeroy',
                fillcolor='rgba(239, 85, 59, 0.2)',
                line=dict(color='rgb(239, 85, 59)', width=2)
            )
        )
        
        fig_deaths.update_layout(
            title="Cumulative Lives Lost on Waitlist",
            yaxis_title="Total Deaths",
            xaxis_title="Years from Now",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig_deaths, use_container_width=True)

    with col2:
        # Key statistics
        total_deaths_baseline = cumulative_baseline_deaths.iloc[-1]
        total_deaths_xeno = cumulative_xeno_deaths.iloc[-1]
        lives_saved = total_deaths_baseline - total_deaths_xeno
        
        st.metric(
            "Total Lives That Could Be Saved",
            f"{lives_saved:,.0f}",
            help="Difference in cumulative deaths over projection period"
        )
        
        # Daily impact
        daily_deaths = waitlist_deaths['baseline_deaths'].mean() / 365
        st.metric(
            "Current Daily Deaths on Waitlist",
            f"{daily_deaths:.1f}",
            help="Average number of people dying each day while waiting"
        )

    st.header("Impact on Wait Times")
    fig_wait = go.Figure()

    fig_wait.add_trace(
        go.Scatter(
            x=waitlist_deaths['year'],
            y=waitlist_deaths['baseline_waitlist'],
            name='Without Xenotransplantation',
            line=dict(color='rgb(239, 85, 59)', width=2)
        )
    )

    fig_wait.add_trace(
        go.Scatter(
            x=waitlist_deaths['year'],
            y=waitlist_deaths['waitlist_with_xeno'],
            name='With Xenotransplantation',
            line=dict(color='rgb(99, 110, 250)', width=2)
        )
    )

    fig_wait.update_layout(
        title="Projected Waitlist Size Over Time",
        yaxis_title="Number of Patients Waiting",
        xaxis_title="Years from Now",
        hovermode="x unified"
    )

    st.plotly_chart(fig_wait, use_container_width=True)

    st.header("Closing the Organ Shortage")
    fig_gap = go.Figure()

    # Calculate annual demand
    annual_demand = waitlist_deaths['annual_additions']

    fig_gap.add_trace(
        go.Scatter(
            x=all_transplants['year'],
            y=annual_demand,
            name='Annual Demand',
            line=dict(color='rgb(239, 85, 59)', width=2, dash='dash')
        )
    )

    fig_gap.add_trace(
        go.Scatter(
            x=all_transplants['year'],
            y=all_transplants['Total'],
            name='Supply with Xenotransplantation',
            fill='tonexty',
            fillcolor='rgba(99, 110, 250, 0.2)',
            line=dict(color='rgb(99, 110, 250)', width=2)
        )
    )

    fig_gap.update_layout(
        title="Annual Organ Supply vs Demand",
        yaxis_title="Number of Organs",
        xaxis_title="Years from Now",
        hovermode="x unified"
    )

    st.plotly_chart(fig_gap, use_container_width=True)

    st.header("Economic Impact")
    col1, col2 = st.columns(2)

    with col1:
        # Cost per life saved over time
        fig_cost = go.Figure()
        
        annual_cost_per_life = projections['total_costs'] / waitlist_deaths['lives_saved']
        
        fig_cost.add_trace(
            go.Scatter(
                x=projections['year'],
                y=annual_cost_per_life,
                name='Cost per Life Saved',
                line=dict(color='rgb(99, 110, 250)', width=2)
            )
        )
        
        fig_cost.update_layout(
            title="Cost per Life Saved Over Time",
            yaxis_title="Cost (USD)",
            xaxis_title="Years from Now",
            hovermode="x"
        )
        
        st.plotly_chart(fig_cost, use_container_width=True)

    with col2:
        # QALY comparison
        st.metric(
            "Cost per QALY",
            f"${metrics['cost_per_qaly']:,.0f}",
            help="Cost per Quality Adjusted Life Year"
        )
        
        # Compare to other medical interventions
        st.markdown("""
        **Comparison to Other Life-Saving Interventions:**
        - Kidney Dialysis: $129,000/QALY
        - Heart Transplant: $100,000/QALY
        - Cancer Treatment: $50,000-150,000/QALY
        """)

if __name__ == "__main__":
    main()