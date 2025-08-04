import re
import matplotlib.pyplot as plt
import numpy as np

def parse_log_file(filename):
    """Parse the log file to extract episodes and percentage increases."""
    episodes = []
    percentages = []
    
    with open(filename, 'r') as file:
        for line in file:
            # Look for lines with "new reward" and percentage increase
            match = re.search(r'new reward: [\d.]+\(([+-][\d.]+)%\) at episode (\d+)', line)
            if match:
                percentage = float(match.group(1))
                episode = int(match.group(2))
                episodes.append(episode)
                percentages.append(percentage)
    
    return episodes, percentages

def plot_reward_increases(episodes, percentages):
    """Plot percentage increases versus episodes."""
    # Start from the second new reward (index 1)
    episodes = episodes[1:]
    percentages = percentages[1:]
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # First plot: Individual percentage increases
    ax1.plot(episodes, percentages, 'bo-', linewidth=2, markersize=6, alpha=0.7)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='No improvement')
    
    # Add trend line
    if len(episodes) > 1:
        z = np.polyfit(episodes, percentages, 1)
        p = np.poly1d(z)
        ax1.plot(episodes, p(episodes), "r--", alpha=0.8, label=f'Trend line (slope: {z[0]:.4f})')
    
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Percentage Increase (%)', fontsize=12)
    ax1.set_title('Individual Reward Improvement Percentage vs Episode\n(Starting from Second New Reward)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add statistics to first plot
    avg_increase = np.mean(percentages)
    max_increase = np.max(percentages)
    min_increase = np.min(percentages)
    
    stats_text = f'Average increase: {avg_increase:.1f}%\nMax increase: {max_increase:.1f}%\nMin increase: {min_increase:.1f}%'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Second plot: Average increase per episode
    # Calculate cumulative average
    cumulative_avg = []
    for i in range(1, len(percentages) + 1):
        cumulative_avg.append(np.mean(percentages[:i]))
    
    ax2.plot(episodes, cumulative_avg, 'go-', linewidth=2, markersize=6, alpha=0.7, label='Cumulative Average')
    ax2.axhline(y=avg_increase, color='orange', linestyle='--', alpha=0.7, label=f'Overall Average: {avg_increase:.1f}%')
    
    # Add trend line for cumulative average
    if len(episodes) > 1:
        z_avg = np.polyfit(episodes, cumulative_avg, 1)
        p_avg = np.poly1d(z_avg)
        ax2.plot(episodes, p_avg(episodes), "orange", linestyle=':', alpha=0.8, label=f'Avg trend (slope: {z_avg[0]:.4f})')
    
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Average Percentage Increase (%)', fontsize=12)
    ax2.set_title('Cumulative Average Reward Improvement vs Episode', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add statistics to second plot
    final_avg = cumulative_avg[-1]
    stats_text2 = f'Final average: {final_avg:.1f}%\nOverall average: {avg_increase:.1f}%\nConvergence trend: {"Stable" if abs(z_avg[0]) < 0.001 else "Still improving" if z_avg[0] > 0 else "Declining"}'
    ax2.text(0.02, 0.98, stats_text2, transform=ax2.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('runs/reward_increases_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return episodes, percentages, cumulative_avg

def main():
    # Parse the log file
    episodes, percentages = parse_log_file('runs/flappybird2.log')
    
    print(f"Found {len(episodes)} new reward entries")
    print(f"Starting from second new reward: {len(episodes)-1} data points")
    
    # Create the plot
    plot_episodes, plot_percentages, cumulative_avg = plot_reward_increases(episodes, percentages)
    
    # Print some statistics
    print(f"\nStatistics (starting from second new reward):")
    print(f"Average percentage increase: {np.mean(plot_percentages):.1f}%")
    print(f"Maximum percentage increase: {np.max(plot_percentages):.1f}%")
    print(f"Minimum percentage increase: {np.min(plot_percentages):.1f}%")
    print(f"Standard deviation: {np.std(plot_percentages):.1f}%")
    print(f"Final cumulative average: {cumulative_avg[-1]:.1f}%")

if __name__ == "__main__":
    main() 