package dicegame;

// The 3 by 11 score table.
// Column numbers run from 2 to 12, so column c is stored at index c - 2.
// A null cell has not been chosen yet. A fixed array fits this table
// because the size never changes: three players and eleven columns.
public class ScoreSheet {

    static final int FIRST_COLUMN = 2;
    static final int LAST_COLUMN = 12;
    static final int COLUMN_COUNT = 11;

    private final ScoreCell[][] cells;
    private final ColumnResult[] results;

    ScoreSheet(int playerCount) {
        cells = new ScoreCell[playerCount][COLUMN_COUNT];
        results = new ColumnResult[COLUMN_COUNT];
    }

    boolean isFilled(int playerIndex, int column) {
        return cells[playerIndex][index(column)] != null;
    }

    void place(int playerIndex, int column, int total) {
        int columnIndex = index(column);
        if (cells[playerIndex][columnIndex] != null) {
            throw new IllegalStateException("Column " + column + " is already filled.");
        }
        cells[playerIndex][columnIndex] = new ScoreCell(total);
    }

    // Called after every placement. Returns null until all three players
    // have an entry, and returns null again if this column was already scored.
    ColumnResult resolveIfComplete(int column, Player[] players) {
        int columnIndex = index(column);
        if (results[columnIndex] != null) {
            return null;
        }
        for (int player = 0; player < cells.length; player++) {
            if (cells[player][columnIndex] == null) {
                return null;
            }
        }

        int[] totals = new int[players.length];
        int best = 0;
        for (int player = 0; player < players.length; player++) {
            totals[player] = cells[player][columnIndex].getTotal();
            if (player == 0 || totals[player] > best) {
                best = totals[player];
            }
        }

        // A single leader wins the column header as points.
        // Two or more players sharing that total is a tie, so the column
        // scores nothing. Anyone strictly below the best total has been
        // beaten and is crossed out, including when the lead is shared.
        int leaders = 0;
        int winner = -1;
        for (int player = 0; player < players.length; player++) {
            if (totals[player] == best) {
                leaders++;
                winner = player;
            } else {
                cells[player][columnIndex].strike();
            }
        }

        boolean awarded = leaders == 1;
        if (awarded) {
            players[winner].addScore(column);
        } else {
            winner = -1;
        }

        ColumnResult result = new ColumnResult(column, best, totals, awarded, winner);
        results[columnIndex] = result;
        return result;
    }

    ColumnResult getResult(int column) {
        return results[index(column)];
    }

    String render(Player[] players) {
        StringBuilder table = new StringBuilder();
        table.append(pad("", 10));
        for (int column = FIRST_COLUMN; column <= LAST_COLUMN; column++) {
            table.append(" | ");
            table.append(padLeft(Integer.toString(column), 6));
        }
        table.append(" | ");
        table.append(padLeft("Scores", 6));
        table.append(System.lineSeparator());

        table.append(repeat("-", 10));
        for (int column = FIRST_COLUMN; column <= LAST_COLUMN; column++) {
            table.append("-+-");
            table.append(repeat("-", 6));
        }
        table.append("-+-");
        table.append(repeat("-", 6));
        table.append(System.lineSeparator());

        for (int player = 0; player < players.length; player++) {
            table.append(pad(players[player].getName(), 10));
            for (int column = FIRST_COLUMN; column <= LAST_COLUMN; column++) {
                table.append(" | ");
                table.append(padLeft(cellText(player, column), 6));
            }
            table.append(" | ");
            table.append(padLeft(Integer.toString(players[player].getScore()), 6));
            table.append(System.lineSeparator());
        }

        table.append(repeat("-", 10));
        for (int column = FIRST_COLUMN; column <= LAST_COLUMN; column++) {
            table.append("-+-");
            table.append(repeat("-", 6));
        }
        table.append("-+-");
        table.append(repeat("-", 6));
        table.append(System.lineSeparator());
        return table.toString();
    }

    private String cellText(int playerIndex, int column) {
        ScoreCell cell = cells[playerIndex][index(column)];
        if (cell == null) {
            return "";
        }
        if (cell.isStruck()) {
            return "X";
        }
        return Integer.toString(cell.getTotal());
    }

    private int index(int column) {
        if (column < FIRST_COLUMN || column > LAST_COLUMN) {
            throw new IllegalArgumentException("Column is outside 2 to 12: " + column);
        }
        return column - FIRST_COLUMN;
    }

    private String pad(String text, int width) {
        StringBuilder padded = new StringBuilder(text);
        while (padded.length() < width) {
            padded.append(' ');
        }
        return padded.toString();
    }

    private String padLeft(String text, int width) {
        StringBuilder padded = new StringBuilder();
        while (padded.length() + text.length() < width) {
            padded.append(' ');
        }
        padded.append(text);
        return padded.toString();
    }

    private String repeat(String token, int count) {
        StringBuilder text = new StringBuilder();
        for (int i = 0; i < count; i++) {
            text.append(token);
        }
        return text.toString();
    }
}
