const std = @import("std");
const print = std.debug.print;

const eloquence = @import("eloquence");

const DB_PATH = "vectors.elq";

pub fn main() !void {
    const dim = 128;
    const DB = eloquence.VectorDB(dim, .Cosine);
    const MetadataPair = eloquence.MetadataPair;

    var db: DB = undefined;

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    if (std.fs.cwd().access(DB_PATH, .{})) |_| {
        print("Loading...\n", .{});
        db = try DB.load(allocator, DB_PATH);
    } else |_| {
        print("Generating...\n", .{});
        db = DB.init(allocator);
    }
    defer db.deinit();

    var prng = std.Random.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
    const rand = prng.random();

    print("Inserting 5,000 random vectors...\n", .{});
    const empty_metadata: []const MetadataPair = &.{};
    for (0..5_000) |_| {
        var raw: @Vector(dim, f32) = undefined;
        for (0..dim) |j| {
            raw[j] = rand.float(f32) * 2.0 - 1.0;
        }
        _ = try db.add(raw, empty_metadata);
    }

    print("Saving...\n", .{});
    try db.save(DB_PATH);

    var query_raw: @Vector(dim, f32) = db.vectors.items[3999];

    for (0..dim) |j| {
        query_raw[j] += (rand.float(f32) * 0.001);
    }
    print("Searching top 10 neighbors...\n", .{});

    const top_10 = try db.search(query_raw, 10);
    defer allocator.free(top_10);

    for (top_10, 1..) |result, rank| {
        print("Rank {} → id: {}, score: {d:.6}\n", .{ rank, result.id, result.score });
    }
}
