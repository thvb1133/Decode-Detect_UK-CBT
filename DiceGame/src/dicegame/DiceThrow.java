package dicegame;

// The result of throwing both dice on one turn.
public class DiceThrow {

    private final int first;
    private final int second;

    DiceThrow(int first, int second) {
        this.first = first;
        this.second = second;
    }

    int getFirst() {
        return first;
    }

    int getSecond() {
        return second;
    }

    // Column entries store this combined total, which is always from 2 to 12.
    int getTotal() {
        return first + second;
    }

    // Five rows of plain text so the throw is easy to read in the console.
    String diagram() {
        String[] left = face(first);
        String[] right = face(second);
        StringBuilder drawing = new StringBuilder();
        for (int row = 0; row < left.length; row++) {
            drawing.append(left[row]);
            drawing.append("   ");
            drawing.append(right[row]);
            drawing.append(System.lineSeparator());
        }
        return drawing.toString();
    }

    private String[] face(int value) {
        return new String[] {
            "+-----+",
            "|     |",
            "|  " + value + "  |",
            "|     |",
            "+-----+"
        };
    }
}
