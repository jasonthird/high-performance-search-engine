const std = @import("std");

const Widget = struct {
    width: u32,
    pub fn render(self: Widget) u32 {
        return self.width * 2;
    }
};

fn computeTotal(items: []const u32) u32 {
    var t: u32 = 0;
    for (items) |v| t += v;
    return t;
}

test "computeTotal sums" {
    try std.testing.expect(computeTotal(&[_]u32{ 1, 2 }) == 3);
}
