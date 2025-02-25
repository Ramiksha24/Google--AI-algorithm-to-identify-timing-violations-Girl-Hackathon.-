import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def visualize_rtl_analysis(results_df, module_name):
    """
    Create comprehensive visualizations for RTL module analysis.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        Results from RTL module analysis
    module_name : str
        Name of the analyzed module
    """
    # Set up the plotting style
    plt.style.use('seaborn')
    
    # Create a figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f'RTL Module Analysis: {module_name}', fontsize=16)
    
    # 1. Signal Depth Distribution
    sns.histplot(
        data=results_df, 
        x='predicted_depth', 
        hue='criticality', 
        multiple='stack', 
        ax=axs[0, 0]
    )
    axs[0, 0].set_title('Signal Depth Distribution by Criticality')
    axs[0, 0].set_xlabel('Predicted Combinational Depth')
    axs[0, 0].set_ylabel('Number of Signals')
    
    # 2. Criticality Pie Chart
    criticality_counts = results_df['criticality'].value_counts()
    axs[0, 1].pie(
        criticality_counts, 
        labels=criticality_counts.index, 
        autopct='%1.1f%%',
        colors=['red', 'orange', 'green']
    )
    axs[0, 1].set_title('Signal Criticality Distribution')
    
    # 3. Fanin vs Fanout Scatter Plot
    scatter = axs[1, 0].scatter(
        results_df['fanin'], 
        results_df['fanout'], 
        c=results_df['predicted_depth'], 
        cmap='viridis', 
        alpha=0.7
    )
    axs[1, 0].set_title('Fanin vs Fanout Colored by Depth')
    axs[1, 0].set_xlabel('Fan-in')
    axs[1, 0].set_ylabel('Fan-out')
    plt.colorbar(scatter, ax=axs[1, 0], label='Predicted Depth')
    
    # 4. Boxplot of Depth by Signal Type
    sns.boxplot(
        x='signal_type', 
        y='predicted_depth', 
        data=results_df, 
        ax=axs[1, 1]
    )
    axs[1, 1].set_title('Predicted Depth by Signal Type')
    axs[1, 1].set_xlabel('Signal Type')
    axs[1, 1].set_ylabel('Predicted Depth')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'{module_name}_analysis_visualization.png', dpi=300)
    plt.close()

def generate_detailed_report(results_df, module_name):
    """
    Generate a detailed textual report of the analysis.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        Results from RTL module analysis
    module_name : str
        Name of the analyzed module
    
    Returns:
    --------
    str
        Detailed analysis report
    """
    report = f"Detailed Analysis Report for {module_name}\n"
    report += "=" * 50 + "\n\n"
    
    # Overall Statistics
    report += "Overall Statistics:\n"
    report += f"Total Signals: {len(results_df)}\n"
    report += f"Average Predicted Depth: {results_df['predicted_depth'].mean():.2f}\n"
    report += f"Max Predicted Depth: {results_df['predicted_depth'].max():.2f}\n"
    report += f"Min Predicted Depth: {results_df['predicted_depth'].min():.2f}\n\n"
    
    # Criticality Breakdown
    report += "Criticality Breakdown:\n"
    criticality_summary = results_df['criticality'].value_counts()
    for criticality, count in criticality_summary.items():
        report += f"{criticality} Criticality Signals: {count} ({count/len(results_df)*100:.1f}%)\n"
    report += "\n"
    
    # Top Critical Signals
    report += "Top 5 Most Critical Signals:\n"
    top_critical = results_df.nlargest(5, 'predicted_depth')
    for _, signal in top_critical.iterrows():
        report += (f"Signal: {signal['signal_name']}\n"
                   f"  Depth: {signal['predicted_depth']:.2f}\n"
                   f"  Criticality: {signal['criticality']}\n"
                   f"  Fan-in: {signal['fanin']}, Fan-out: {signal['fanout']}\n\n")
    
    return report

def main():
    """
    Example usage of RTL analysis and visualization.
    """
    from rtl_analysis import analyze_rtl_module, generate_timing_report
    
    # Example ALU RTL
    alu_rtl = """
    module alu_32bit (
        input wire clk,
        input wire rst,
        input wire [31:0] a,
        input wire [31:0] b,
        input wire [3:0] op,
        output reg [31:0] result,
        output reg overflow,
        output reg zero_flag
    );
        reg [31:0] temp_result;
        wire carry_out;
        
        always @(*) begin
            case(op)
                4'b0000: temp_result = a + b;
                4'b0001: temp_result = a - b;
                4'b0010: temp_result = a & b;
                4'b0011: temp_result = a | b;
                4'b0100: temp_result = a ^ b;
                4'b0101: temp_result = ~a;
                4'b0110: temp_result = a << b[4:0];
                4'b0111: temp_result = a >> b[4:0];
                default: temp_result = 32'h0;
            endcase
        end
        
        always @(posedge clk or posedge rst) begin
            if (rst) begin
                result <= 32'h0;
                overflow <= 1'b0;
                zero_flag <= 1'b0;
            end else begin
                result <= temp_result;
                overflow <= (op == 4'b0000 && (a[31] == b[31]) && (temp_result[31] != a[31])) || 
                           (op == 4'b0001 && (a[31] != b[31]) && (temp_result[31] != a[31]));
                zero_flag <= (temp_result == 32'h0);
            end
        end
    endmodule
    """
    
    # Analyze module
    results = analyze_rtl_module(alu_rtl)
    
    # Create visualizations
    visualize_rtl_analysis(results, "alu_32bit")
    
    # Generate detailed report
    detailed_report = generate_detailed_report(results, "alu_32bit")
    print(detailed_report)
    
    # Generate timing report
    timing_report = generate_timing_report(results, "alu_32bit", clock_period_ns=2.0)
    print("\nTiming Report:")
    print(timing_report)

if __name__ == "__main__":
    main()
