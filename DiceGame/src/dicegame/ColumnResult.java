package dicegame;

// What happened when the third player filled a column.
public class ColumnResult {

    private final int column;
    private final int bestTotal;
    private final int[] totals;
    private final boolean awarded;
    private final int winnerIndex;

    ColumnResult(int column, int bestTotal, int[] totals, boolean awarded, int winnerIndex) {
        this.column = column;
        this.bestTotal = bestTotal;
        this.totals = totals;
        this.awarded = awarded;
        this.winnerIndex = winnerIndex;
    }

    int getColumn() {
        return column;
    }

    boolean isAwarded() {
        return awarded;
    }

    int getWinnerIndex() {
        return winnerIndex;
    }

    // Spoken when the column is filled, before the updated table is shown.
    String liveMessage(Player[] players) {
        if (awarded) {
            return "Column " + column + " is complete. "
                    + players[winnerIndex].getName() + " wins it with "
                    + bestTotal + " and scores " + column + " points. "
                    + describeBeaten(players) + ".";
        }

        String leaders = describeLeaders(players);
        String message = "Column " + column + " is complete. It is a tie on "
                + bestTotal + " between " + leaders + ", so nobody scores.";
        if (countBelowBest() > 0) {
            message = message + " " + describeBeaten(players) + ".";
        }
        return message;
    }

    // One line in the closing summary. Same facts as the live message.
    String summaryLine(Player[] players) {
        if (awarded) {
            return "Column " + width2(column) + ": "
                    + players[winnerIndex].getName() + " won with "
                    + bestTotal + " and scored " + column + " points. "
                    + describeBeaten(players) + ".";
        }

        String line = "Column " + width2(column) + ": tie on " + bestTotal
                + " between " + describeLeaders(players)
                + ". No points.";
        if (countBelowBest() > 0) {
            line = line + " " + describeBeaten(players) + ".";
        }
        return line;
    }

    private String describeLeaders(Player[] players) {
        String[] names = new String[totals.length];
        int count = 0;
        for (int i = 0; i < totals.length; i++) {
            if (totals[i] == bestTotal) {
                names[count] = players[i].getName();
                count++;
            }
        }
        return join(names, count);
    }

    private String describeBeaten(Player[] players) {
        String[] names = new String[totals.length];
        int count = 0;
        for (int i = 0; i < totals.length; i++) {
            if (totals[i] < bestTotal) {
                names[count] = players[i].getName() + " (" + totals[i] + ")";
                count++;
            }
        }
        if (count == 1) {
            return names[0] + " is crossed out";
        }
        return join(names, count) + " are crossed out";
    }

    private int countBelowBest() {
        int count = 0;
        for (int i = 0; i < totals.length; i++) {
            if (totals[i] < bestTotal) {
                count++;
            }
        }
        return count;
    }

    private String join(String[] names, int count) {
        if (count == 1) {
            return names[0];
        }
        if (count == 2) {
            return names[0] + " and " + names[1];
        }
        StringBuilder text = new StringBuilder();
        for (int i = 0; i < count; i++) {
            if (i == count - 1) {
                text.append(" and ");
            } else if (i > 0) {
                text.append(", ");
            }
            text.append(names[i]);
        }
        return text.toString();
    }

    private String width2(int value) {
        if (value < 10) {
            return " " + value;
        }
        return Integer.toString(value);
    }
}
