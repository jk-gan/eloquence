const std = @import("std");
const assert = std.debug.assert;
const meta = @import("metadata.zig");
const MetadataPair = meta.MetadataPair;

pub const Header = extern struct {
    magic: u32,
    version: u32,
    dim: u32,
    _padding: u32,
    count: u64,
    offset_table_pos: u64,
    blob_pos: u64,

    comptime {
        assert(@sizeOf(Header) == 40);
        assert(@alignOf(Header) == 8);
    }
};

pub const OffsetEntry = extern struct {
    offset: u64,
    length: u32,
    _padding: u32,

    comptime {
        assert(@sizeOf(OffsetEntry) == 16);
        assert(@alignOf(OffsetEntry) == 8);
    }
};

// "ELQ" + 0x01 (in hex) = 0x00514C45
pub const MAGIC_NUMBER: u32 = 0x514C4501;

pub fn serialize_metadata(writer: anytype, metadata: []const MetadataPair) !void {
    try writer.writeInt(u16, @intCast(metadata.len), .little);

    for (metadata) |pair| {
        try writer.writeInt(u8, @intCast(pair.key.len), .little);
        try writer.writeAll(pair.key);

        // value_type: 0 => boolean, 1 => integer, 2 => string
        switch (pair.value) {
            .boolean => |b| {
                try writer.writeInt(u8, 0, .little);
                try writer.writeInt(u8, @intFromBool(b), .little);
            },
            .integer => |i| {
                try writer.writeInt(u8, 1, .little);
                try writer.writeInt(i64, i, .little);
            },
            .string => |s| {
                try writer.writeInt(u8, 2, .little);
                try writer.writeInt(u16, @intCast(s.len), .little);
                try writer.writeAll(s);
            },
        }
    }
}

pub fn deserialize_metadata(reader: anytype, allocator: std.mem.Allocator) ![]const MetadataPair {
    const pair_count = try reader.takeInt(u16, .little);

    const pairs = try allocator.alloc(MetadataPair, pair_count);

    for (pairs) |*pair| {
        const key_len = try reader.takeInt(u8, .little);
        const key = try allocator.alloc(u8, key_len);
        try reader.readSliceAll(key);
        pair.key = key;

        const value_type = try reader.takeInt(u8, .little);
        pair.value = switch (value_type) {
            0 => .{ .boolean = try reader.takeInt(u8, .little) != 0 },
            1 => .{ .integer = try reader.takeInt(i64, .little) },
            2 => blk: {
                const str_len = try reader.takeInt(u16, .little);
                const str = try allocator.alloc(u8, str_len);
                try reader.readSliceAll(str);
                break :blk .{ .string = str };
            },
            else => return error.InvalidMetadataValueType,
        };
    }

    return pairs;
}
